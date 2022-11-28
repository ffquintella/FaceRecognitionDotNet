using System;
using System.Collections.Generic;
using System.Linq;
using DlibDotNet;
using DlibDotNet.Dnn;

namespace FaceRecognitionDotNet.Dlib.Python
{
    public interface IFaceRecognitionModel
    {
        Matrix<double> ComputeFaceDescriptor(LossMetric net, Image img, FullObjectDetection face,
            int numJitters);

        IEnumerable<Matrix<double>> ComputeFaceDescriptors(LossMetric net, Image img,
            IEnumerable<FullObjectDetection> faces, int numJitters);

        IEnumerable<IEnumerable<Matrix<double>>> BatchComputeFaceDescriptors(LossMetric net,
            IList<Image> batchImages,
            IList<IEnumerable<FullObjectDetection>> batchFaces,
            int numJitters);
        
        
    }
}