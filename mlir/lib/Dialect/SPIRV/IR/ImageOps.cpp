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

static LogicalResult verifyImageOperands(Operation *imageOp,
                                         spirv::ImageOperandsAttr attr,
                                         Operation::operand_range operands) {
  if (!attr) {
    if (operands.empty())
      return success();

    return imageOp->emitError("the Image Operands should encode what operands "
                              "follow, as per Image Operands");
  }

  // TODO: Add the validation rules for the following Image Operands.
  spirv::ImageOperands noSupportOperands =
      spirv::ImageOperands::Bias | spirv::ImageOperands::Lod |
      spirv::ImageOperands::Grad | spirv::ImageOperands::ConstOffset |
      spirv::ImageOperands::Offset | spirv::ImageOperands::ConstOffsets |
      spirv::ImageOperands::Sample | spirv::ImageOperands::MinLod |
      spirv::ImageOperands::MakeTexelAvailable |
      spirv::ImageOperands::MakeTexelVisible |
      spirv::ImageOperands::SignExtend | spirv::ImageOperands::ZeroExtend;

  assert(!spirv::bitEnumContainsAny(attr.getValue(), noSupportOperands) &&
         "unimplemented operands of Image Operands");

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
