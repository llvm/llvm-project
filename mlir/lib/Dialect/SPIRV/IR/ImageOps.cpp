//===- ImageOps.cpp - MLIR SPIR-V Image Ops  ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the image operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Common utility functions
//===----------------------------------------------------------------------===//

// TODO: In the future we should model image operands better, so we can move
// some verification into ODS.
static LogicalResult verifyImageOperands(Operation *imageOp,
                                         spirv::ImageOperandsAttr attr,
                                         Operation::operand_range operands) {
  if (!attr) {
    if (operands.empty())
      return success();

    return imageOp->emitError("the Image Operands should encode what operands "
                              "follow, as per Image Operands");
  }

  if (spirv::bitEnumContainsAll(attr.getValue(),
                                spirv::ImageOperands::Lod |
                                    spirv::ImageOperands::Grad))
    return imageOp->emitError(
        "it is invalid to set both the Lod and Grad bits");

  size_t index = 0;

  // The order we process operands is important. In case of multiple argument
  // taking operands, the arguments are ordered starting with operands having
  // smaller-numbered bits first.
  if (spirv::bitEnumContainsAny(attr.getValue(), spirv::ImageOperands::Bias)) {
    if (!isa<spirv::ImplicitLodOpInterface>(imageOp))
      return imageOp->emitError(
          "Bias is only valid with implicit-lod instructions");

    if (index + 1 > operands.size())
      return imageOp->emitError("Bias operand requires 1 argument");

    if (!isa<FloatType>(operands[index].getType()))
      return imageOp->emitError("Bias must be a floating-point type scalar");

    auto samplingOp = cast<spirv::SamplingOpInterface>(imageOp);
    auto sampledImageType =
        cast<spirv::SampledImageType>(samplingOp.getSampledImage().getType());
    auto imageType = cast<spirv::ImageType>(sampledImageType.getImageType());

    if (!llvm::is_contained({spirv::Dim::Dim1D, spirv::Dim::Dim2D,
                             spirv::Dim::Dim3D, spirv::Dim::Cube},
                            imageType.getDim()))
      return imageOp->emitError(
          "Bias must only be used with an image type that has "
          "a dim operand of 1D, 2D, 3D, or Cube");

    if (imageType.getSamplingInfo() != spirv::ImageSamplingInfo::SingleSampled)
      return imageOp->emitError("Bias must only be used with an image type "
                                "that has a MS operand of 0");

    ++index;
  }

  if (spirv::bitEnumContainsAny(attr.getValue(), spirv::ImageOperands::Lod)) {
    if (!isa<spirv::ExplicitLodOpInterface>(imageOp) &&
        !isa<spirv::FetchOpInterface>(imageOp))
      return imageOp->emitError(
          "Lod is only valid with explicit-lod and fetch instructions");

    if (index + 1 > operands.size())
      return imageOp->emitError("Lod operand requires 1 argument");

    spirv::ImageType imageType;

    if (isa<spirv::SamplingOpInterface>(imageOp)) {
      if (!isa<mlir::FloatType>(operands[index].getType()))
        return imageOp->emitError("for sampling operations, Lod must be a "
                                  "floating-point type scalar");

      auto samplingOp = cast<spirv::SamplingOpInterface>(imageOp);
      auto sampledImageType = llvm::cast<spirv::SampledImageType>(
          samplingOp.getSampledImage().getType());
      imageType = cast<spirv::ImageType>(sampledImageType.getImageType());
    } else {
      if (!isa<mlir::IntegerType>(operands[index].getType()))
        return imageOp->emitError(
            "for fetch operations, Lod must be an integer type scalar");

      auto fetchOp = cast<spirv::FetchOpInterface>(imageOp);
      imageType = cast<spirv::ImageType>(fetchOp.getImage().getType());
    }

    if (!llvm::is_contained({spirv::Dim::Dim1D, spirv::Dim::Dim2D,
                             spirv::Dim::Dim3D, spirv::Dim::Cube},
                            imageType.getDim()))
      return imageOp->emitError(
          "Lod must only be used with an image type that has "
          "a dim operand of 1D, 2D, 3D, or Cube");

    if (imageType.getSamplingInfo() != spirv::ImageSamplingInfo::SingleSampled)
      return imageOp->emitError("Lod must only be used with an image type that "
                                "has a MS operand of 0");

    ++index;
  }

  if (spirv::bitEnumContainsAny(attr.getValue(), spirv::ImageOperands::Grad)) {
    if (!isa<spirv::ExplicitLodOpInterface>(imageOp))
      return imageOp->emitError(
          "Grad is only valid with explicit-lod instructions");

    if (index + 2 > operands.size())
      return imageOp->emitError(
          "Grad operand requires 2 arguments (scalars or vectors)");

    auto samplingOp = cast<spirv::SamplingOpInterface>(imageOp);
    auto sampledImageType =
        cast<spirv::SampledImageType>(samplingOp.getSampledImage().getType());
    auto imageType = cast<spirv::ImageType>(sampledImageType.getImageType());

    if (imageType.getSamplingInfo() != spirv::ImageSamplingInfo::SingleSampled)
      return imageOp->emitError("Grad must only be used with an image type "
                                "that has a MS operand of 0");

    int64_t numberOfComponents = 0;

    auto coordVector =
        dyn_cast<mlir::VectorType>(samplingOp.getCoordinate().getType());
    if (coordVector) {
      numberOfComponents = coordVector.getNumElements();
      if (imageType.getArrayedInfo() == spirv::ImageArrayedInfo::Arrayed)
        numberOfComponents -= 1;
    } else {
      numberOfComponents = 1;
    }

    assert(numberOfComponents > 0);

    auto dXVector = dyn_cast<mlir::VectorType>(operands[index].getType());
    auto dYVector = dyn_cast<mlir::VectorType>(operands[index + 1].getType());
    if (dXVector && dYVector) {
      if (dXVector.getNumElements() != dYVector.getNumElements() ||
          dXVector.getNumElements() != numberOfComponents)
        return imageOp->emitError(
            "number of components of each Grad argument must equal the number "
            "of components in coordinate, minus the array layer component, if "
            "present");

      if (!isa<mlir::FloatType>(dXVector.getElementType()) ||
          !isa<mlir::FloatType>(dYVector.getElementType()))
        return imageOp->emitError(
            "Grad arguments must be a vector of floating-point type");
    } else if (isa<mlir::FloatType>(operands[index].getType()) &&
               isa<mlir::FloatType>(operands[index + 1].getType())) {
      if (numberOfComponents != 1)
        return imageOp->emitError(
            "number of components of each Grad argument must equal the number "
            "of components in coordinate, minus the array layer component, if "
            "present");
    } else {
      return imageOp->emitError(
          "Grad arguments must be a scalar or vector of floating-point type");
    }

    index += 2;
  }

  // TODO: Add the validation rules for the following Image Operands.
  spirv::ImageOperands noSupportOperands =
      spirv::ImageOperands::ConstOffset | spirv::ImageOperands::Offset |
      spirv::ImageOperands::ConstOffsets | spirv::ImageOperands::Sample |
      spirv::ImageOperands::MinLod | spirv::ImageOperands::MakeTexelAvailable |
      spirv::ImageOperands::MakeTexelVisible |
      spirv::ImageOperands::SignExtend | spirv::ImageOperands::ZeroExtend;

  assert(!spirv::bitEnumContainsAny(attr.getValue(), noSupportOperands) &&
         "unimplemented operands of Image Operands");
  (void)noSupportOperands;

  if (index < operands.size())
    return imageOp->emitError(
        "too many image operand arguments have been provided");

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.ImageDrefGather
//===----------------------------------------------------------------------===//

LogicalResult spirv::ImageDrefGatherOp::verify() {
  return verifyImageOperands(getOperation(), getImageOperandsAttr(),
                             getOperandArguments());
}

//===----------------------------------------------------------------------===//
// spirv.ImageReadOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::ImageReadOp::verify() {
  // TODO: Do we need check for: "If the Arrayed operand is 1, then additional
  // capabilities may be required; e.g., ImageCubeArray, or ImageMSArray."?

  // TODO: Ideally it should be somewhere verified that "If the Image Dim
  // operand is not SubpassData, the Image Format must not be Unknown, unless
  // the StorageImageReadWithoutFormat Capability was declared." This function
  // however may not be the suitable place for such verification.

  return verifyImageOperands(getOperation(), getImageOperandsAttr(),
                             getOperandArguments());
}

//===----------------------------------------------------------------------===//
// spirv.ImageWriteOp
//===----------------------------------------------------------------------===//

LogicalResult spirv::ImageWriteOp::verify() {
  // TODO: Do we need check for: "If the Arrayed operand is 1, then additional
  // capabilities may be required; e.g., ImageCubeArray, or ImageMSArray."?

  // TODO: Ideally it should be somewhere verified that "The Image Format must
  // not be Unknown, unless the StorageImageWriteWithoutFormat Capability was
  // declared." This function however may not be the suitable place for such
  // verification.

  return verifyImageOperands(getOperation(), getImageOperandsAttr(),
                             getOperandArguments());
}

//===----------------------------------------------------------------------===//
// spirv.ImageQuerySize
//===----------------------------------------------------------------------===//

LogicalResult spirv::ImageQuerySizeOp::verify() {
  spirv::ImageType imageType =
      llvm::cast<spirv::ImageType>(getImage().getType());
  Type resultType = getResult().getType();

  spirv::Dim dim = imageType.getDim();
  spirv::ImageSamplingInfo samplingInfo = imageType.getSamplingInfo();
  spirv::ImageSamplerUseInfo samplerInfo = imageType.getSamplerUseInfo();
  switch (dim) {
  case spirv::Dim::Dim1D:
  case spirv::Dim::Dim2D:
  case spirv::Dim::Dim3D:
  case spirv::Dim::Cube:
    if (samplingInfo != spirv::ImageSamplingInfo::MultiSampled &&
        samplerInfo != spirv::ImageSamplerUseInfo::SamplerUnknown &&
        samplerInfo != spirv::ImageSamplerUseInfo::NoSampler)
      return emitError(
          "if Dim is 1D, 2D, 3D, or Cube, "
          "it must also have either an MS of 1 or a Sampled of 0 or 2");
    break;
  case spirv::Dim::Buffer:
  case spirv::Dim::Rect:
    break;
  default:
    return emitError("the Dim operand of the image type must "
                     "be 1D, 2D, 3D, Buffer, Cube, or Rect");
  }

  unsigned componentNumber = 0;
  switch (dim) {
  case spirv::Dim::Dim1D:
  case spirv::Dim::Buffer:
    componentNumber = 1;
    break;
  case spirv::Dim::Dim2D:
  case spirv::Dim::Cube:
  case spirv::Dim::Rect:
    componentNumber = 2;
    break;
  case spirv::Dim::Dim3D:
    componentNumber = 3;
    break;
  default:
    break;
  }

  if (imageType.getArrayedInfo() == spirv::ImageArrayedInfo::Arrayed)
    componentNumber += 1;

  unsigned resultComponentNumber = 1;
  if (auto resultVectorType = llvm::dyn_cast<VectorType>(resultType))
    resultComponentNumber = resultVectorType.getNumElements();

  if (componentNumber != resultComponentNumber)
    return emitError("expected the result to have ")
           << componentNumber << " component(s), but found "
           << resultComponentNumber << " component(s)";

  return success();
}

//===----------------------------------------------------------------------===//
// spirv.ImageSampleImplicitLod
//===----------------------------------------------------------------------===//

LogicalResult spirv::ImageSampleImplicitLodOp::verify() {
  return verifyImageOperands(getOperation(), getImageOperandsAttr(),
                             getOperandArguments());
}

//===----------------------------------------------------------------------===//
// spirv.ImageSampleExplicitLod
//===----------------------------------------------------------------------===//

LogicalResult spirv::ImageSampleExplicitLodOp::verify() {
  // TODO: It should be verified somewhere that: "Unless the Kernel capability
  // is declared, it [Coordinate] must be floating point."

  return verifyImageOperands(getOperation(), getImageOperandsAttr(),
                             getOperandArguments());
}

//===----------------------------------------------------------------------===//
// spirv.ImageSampleProjDrefImplicitLod
//===----------------------------------------------------------------------===//

LogicalResult spirv::ImageSampleProjDrefImplicitLodOp::verify() {
  return verifyImageOperands(getOperation(), getImageOperandsAttr(),
                             getOperandArguments());
}
