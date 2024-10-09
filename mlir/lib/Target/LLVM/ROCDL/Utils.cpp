//===- Utils.cpp - MLIR ROCDL target utils ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files defines ROCDL target related utility classes and functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Frontend/Offloading/Utility.h"

using namespace mlir;
using namespace mlir::ROCDL;

std::optional<DenseMap<StringAttr, NamedAttrList>>
mlir::ROCDL::getAMDHSAKernelsELFMetadata(Builder &builder,
                                         ArrayRef<char> elfData) {
  uint16_t elfABIVersion;
  llvm::StringMap<llvm::offloading::amdgpu::AMDGPUKernelMetaData> kernels;
  llvm::MemoryBufferRef buffer(StringRef(elfData.data(), elfData.size()),
                               "buffer");
  // Get the metadata.
  llvm::Error error = llvm::offloading::amdgpu::getAMDGPUMetaDataFromImage(
      buffer, kernels, elfABIVersion);
  // Return `nullopt` if the metadata couldn't be retrieved.
  if (error) {
    llvm::consumeError(std::move(error));
    return std::nullopt;
  }
  // Helper lambda for converting values.
  auto getI32Array = [&builder](const uint32_t *array) {
    return builder.getDenseI32ArrayAttr({static_cast<int32_t>(array[0]),
                                         static_cast<int32_t>(array[1]),
                                         static_cast<int32_t>(array[2])});
  };
  DenseMap<StringAttr, NamedAttrList> kernelMD;
  for (const auto &[name, kernel] : kernels) {
    NamedAttrList attrs;
    // Add kernel metadata.
    attrs.append("agpr_count", builder.getI64IntegerAttr(kernel.AGPRCount));
    attrs.append("sgpr_count", builder.getI64IntegerAttr(kernel.SGPRCount));
    attrs.append("vgpr_count", builder.getI64IntegerAttr(kernel.VGPRCount));
    attrs.append("sgpr_spill_count",
                 builder.getI64IntegerAttr(kernel.SGPRSpillCount));
    attrs.append("vgpr_spill_count",
                 builder.getI64IntegerAttr(kernel.VGPRSpillCount));
    attrs.append("wavefront_size",
                 builder.getI64IntegerAttr(kernel.WavefrontSize));
    attrs.append("max_flat_workgroup_size",
                 builder.getI64IntegerAttr(kernel.MaxFlatWorkgroupSize));
    attrs.append("group_segment_fixed_size",
                 builder.getI64IntegerAttr(kernel.GroupSegmentList));
    attrs.append("private_segment_fixed_size",
                 builder.getI64IntegerAttr(kernel.PrivateSegmentSize));
    attrs.append("reqd_workgroup_size",
                 getI32Array(kernel.RequestedWorkgroupSize));
    attrs.append("workgroup_size_hint", getI32Array(kernel.WorkgroupSizeHint));
    kernelMD[builder.getStringAttr(name)] = std::move(attrs);
  }
  return std::move(kernelMD);
}

gpu::KernelTableAttr mlir::ROCDL::getKernelMetadata(Operation *gpuModule,
                                                    ArrayRef<char> elfData) {
  auto module = cast<gpu::GPUModuleOp>(gpuModule);
  Builder builder(module.getContext());
  SmallVector<gpu::KernelMetadataAttr> kernels;
  std::optional<DenseMap<StringAttr, NamedAttrList>> mdMapOrNull =
      getAMDHSAKernelsELFMetadata(builder, elfData);
  for (auto funcOp : module.getBody()->getOps<LLVM::LLVMFuncOp>()) {
    if (!funcOp->getDiscardableAttr("rocdl.kernel"))
      continue;
    kernels.push_back(gpu::KernelMetadataAttr::get(
        funcOp, mdMapOrNull ? builder.getDictionaryAttr(
                                  mdMapOrNull->lookup(funcOp.getNameAttr()))
                            : nullptr));
  }
  return gpu::KernelTableAttr::get(gpuModule->getContext(), kernels);
}
