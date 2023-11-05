//===- Target.cpp - MLIR SPIR-V target compilation --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files defines SPIR-V target related functions including registration
// calls for the `#spirv.target_env` compilation attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/SPIRV/Target.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Target/SPIRV/Serialization.h"

#include <cstdlib>
#include <cstring>

using namespace mlir;
using namespace mlir::spirv;

namespace {
// SPIR-V implementation of the gpu:TargetAttrInterface.
class SPIRVTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<SPIRVTargetAttrImpl> {
public:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute,
                         const SmallVector<char, 0> &object,
                         const gpu::TargetOptions &options) const;
};
} // namespace

// Register the SPIR-V dialect, the SPIR-V translation & the target interface.
void mlir::spirv::registerSPIRVTargetInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, spirv::SPIRVDialect *dialect) {
    spirv::TargetEnvAttr::attachInterface<SPIRVTargetAttrImpl>(*ctx);
  });
}

void mlir::spirv::registerSPIRVTargetInterfaceExternalModels(
    MLIRContext &context) {
  DialectRegistry registry;
  registerSPIRVTargetInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
}

// Reuse from existing serializer
std::optional<SmallVector<char, 0>> SPIRVTargetAttrImpl::serializeToObject(
    Attribute attribute, Operation *module,
    const gpu::TargetOptions &options) const {
  if (!module)
    return std::nullopt;
  auto gpuMod = dyn_cast<gpu::GPUModuleOp>(module);
  if (!gpuMod) {
    module->emitError("expected to be a gpu.module op");
    return std::nullopt;
  }
  auto spvMods = gpuMod.getOps<spirv::ModuleOp>();
  if (spvMods.empty())
    return std::nullopt;

  auto spvMod = *spvMods.begin();
  llvm::SmallVector<uint32_t, 0> spvBinary;

  spvBinary.clear();
  // Serialize the spirv.module op to SPIR-V blob.
  if (mlir::failed(spirv::serialize(spvMod, spvBinary))) {
    spvMod.emitError() << "failed to serialize SPIR-V module";
    return std::nullopt;
  }

  SmallVector<char, 0> spvData(spvBinary.size() * sizeof(uint32_t), 0);
  std::memcpy(spvData.data(), spvBinary.data(), spvData.size());

  spvMod.erase();
  return spvData;
}

// Prepare Attribute for gpu.binary with serialized kernel object
Attribute
SPIRVTargetAttrImpl::createObject(Attribute attribute,
                                  const SmallVector<char, 0> &object,
                                  const gpu::TargetOptions &options) const {
  gpu::CompilationTarget format = options.getCompilationTarget();
  DictionaryAttr objectProps;
  Builder builder(attribute.getContext());
  return builder.getAttr<gpu::ObjectAttr>(
      attribute, format,
      builder.getStringAttr(StringRef(object.data(), object.size())),
      objectProps);
}
