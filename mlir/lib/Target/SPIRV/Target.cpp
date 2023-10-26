//===- Target.cpp - MLIR SPIRV target compilation ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files defines SPIRV target related functions including registration
// calls for the `#spirv.target` compilation attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/SPIRV/Target.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/SPIRV/Serialization.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"

#include <cstdlib>
#include <cstring>

using namespace mlir;
using namespace mlir::spirv;

namespace {
// Implementation of the `TargetAttrInterface` model.
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

// Register the SPIRV dialect, the SPIRV translation & the target interface.
void mlir::spirv::registerSPIRVTargetInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, spirv::SPIRVDialect *dialect) {
    spirv::SPIRVTargetAttr::attachInterface<SPIRVTargetAttrImpl>(*ctx);
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
  assert(module && "The module must be non null.");
  if (!module)
    return std::nullopt;
  if (!mlir::isa<gpu::GPUModuleOp>(module)) {
    module->emitError("Module must be a GPU module.");
    return std::nullopt;
  }
  auto gpuMod = dyn_cast<gpu::GPUModuleOp>(module);
  auto spvMods = gpuMod.getOps<spirv::ModuleOp>();
  // Empty spirv::ModuleOp
  if (spvMods.empty()) {
    return std::nullopt;
  }
  auto spvMod = *spvMods.begin();
  llvm::SmallVector<uint32_t, 0> spvBinary;

  spvBinary.clear();
  // serialize the spv module to spv binary
  if (mlir::failed(spirv::serialize(spvMod, spvBinary))) {
    spvMod.emitError() << "Failed to serialize SPIR-V module";
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
  auto target = cast<SPIRVTargetAttr>(attribute);
  gpu::CompilationTarget format = options.getCompilationTarget();
  DictionaryAttr objectProps;
  Builder builder(attribute.getContext());
  return builder.getAttr<gpu::ObjectAttr>(
      attribute, format,
      builder.getStringAttr(StringRef(object.data(), object.size())),
      objectProps);
}
