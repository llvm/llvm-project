//===- OpenACCToLLVMIRTranslation.cpp - Translate OpenACC to LLVM IR ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the OpenACC to LLVM IR translation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/OpenACC/OpenACCToLLVMIRTranslation.h"

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/XeVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace mlir;

namespace {
/// The extracted generic information for a target specific offload attribute.
struct OffloadTargetInfo {
  StringRef triple;
  StringRef arch;
};

/// Returns the offloading information of a GPU target attribute, or
/// `std::nullopt` if the target is not supported.
static std::optional<OffloadTargetInfo> getOffloadTargetInfo(Attribute target) {
  return llvm::TypeSwitch<Attribute, std::optional<OffloadTargetInfo>>(target)
      .Case([](NVVM::NVVMTargetAttr nvvm) {
        return OffloadTargetInfo{nvvm.getTriple(), nvvm.getChip()};
      })
      .Case([](ROCDL::ROCDLTargetAttr rocdl) {
        return OffloadTargetInfo{rocdl.getTriple(), rocdl.getChip()};
      })
      .Case([](xevm::XeVMTargetAttr xevm) {
        return OffloadTargetInfo{xevm.getTriple(), xevm.getChip()};
      })
      .Default([](Attribute) { return std::nullopt; });
}

/// Implementation of the `OffloadingLLVMTranslationAttrInterface` model for
/// `#acc.offload_packager`.
class OffloadPackagerAttrImpl
    : public gpu::OffloadingLLVMTranslationAttrInterface::FallbackModel<
          OffloadPackagerAttrImpl> {
public:
  /// Embeds the objects of a `gpu.binary` in the host module as offloading
  /// binaries placed in the `.llvm.offloading` section.
  LogicalResult embedBinary(Attribute attribute, Operation *operation,
                            llvm::IRBuilderBase &builder,
                            LLVM::ModuleTranslation &moduleTranslation) const;

  LogicalResult launchKernel(Attribute attribute,
                             Operation *launchFuncOperation,
                             Operation *binaryOperation,
                             llvm::IRBuilderBase &builder,
                             LLVM::ModuleTranslation &moduleTranslation) const;
};
} // namespace

LogicalResult OffloadPackagerAttrImpl::embedBinary(
    Attribute attribute, Operation *operation, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) const {
  assert(operation && "The binary operation must be non null.");
  auto op = mlir::dyn_cast<gpu::BinaryOp>(operation);
  if (!op) {
    operation->emitError("operation must be a GPU binary");
    return failure();
  }

  using namespace llvm::object;

  SmallVector<char> binaryData;
  llvm::raw_svector_ostream os(binaryData);
  for (Attribute attr : op.getObjectsAttr().getValue()) {
    auto object = dyn_cast<gpu::ObjectAttr>(attr);
    if (!object)
      return op.emitError("expected a gpu.object");

    // The section is chosen by the offloading binary, so a requested one would
    // be ignored.
    if (DictionaryAttr properties = object.getProperties())
      if (properties.get(gpu::elfSectionName))
        return op.emitError("the offload packager does not support custom "
                            "sections");

    std::optional<OffloadTargetInfo> target =
        getOffloadTargetInfo(object.getTarget());
    if (!target)
      return op.emitError("the offload packager does not support the target ")
             << object.getTarget();

    OffloadBinary::OffloadingImage image;
    bool requiresNullTerminator = false;
    switch (object.getFormat()) {
    case gpu::CompilationTarget::Offload:
      image.TheImageKind = IMG_Bitcode;
      requiresNullTerminator = true;
      break;
    case gpu::CompilationTarget::Binary:
      image.TheImageKind = IMG_Object;
      requiresNullTerminator = false;
      break;
    case gpu::CompilationTarget::Assembly:
    case gpu::CompilationTarget::Fatbin:
      return op.emitError("the offloading linker wrapper cannot link the "
                          "assembly or fatbin formats");
    }

    StringRef serialized = object.getObject().getValue();
    image.Image = llvm::MemoryBuffer::getMemBuffer(
        serialized, /*BufferName=*/"", requiresNullTerminator);
    image.TheOffloadKind = OFK_OpenACC;
    image.StringData["arch"] = target->arch;
    image.StringData["triple"] = target->triple;

    llvm::SmallString<0> buffer = OffloadBinary::write(image);
    if (buffer.size() % OffloadBinary::getAlignment() != 0)
      return op.emitError("offload binary has invalid size alignment");
    os << buffer;
  }

  llvm::Module &m = *moduleTranslation.getLLVMModule();
  llvm::Constant *serializedCst = llvm::ConstantDataArray::getString(
      m.getContext(), os.str(), /*AddNull=*/false);
  auto *serializedObj =
      new llvm::GlobalVariable(m, serializedCst->getType(), /*isConstant=*/true,
                               llvm::GlobalValue::InternalLinkage,
                               serializedCst, op.getName() + "_binary");
  serializedObj->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
  serializedObj->setSection(".llvm.offloading");
  serializedObj->setAlignment(llvm::Align(OffloadBinary::getAlignment()));
  llvm::appendToUsed(m, {serializedObj});
  return success();
}

LogicalResult
OffloadPackagerAttrImpl::launchKernel(Attribute, Operation *launchFuncOperation,
                                      Operation *, llvm::IRBuilderBase &,
                                      LLVM::ModuleTranslation &) const {
  return launchFuncOperation->emitError("unexpected kernel launch");
}

/// Registers the host function or variable `op` as an offloading entry of kind
/// OpenACC.
static LogicalResult
convertOffloadEntry(Operation *op, LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Module *m = moduleTranslation.getLLVMModule();
  llvm::GlobalValue *global = nullptr;
  uint64_t size = 0;
  if (auto func = dyn_cast<LLVM::LLVMFuncOp>(op)) {
    global = moduleTranslation.lookupFunction(func.getName());
    assert(global && "could not find the translated host symbol");
  } else if (auto variable = dyn_cast<LLVM::GlobalOp>(op)) {
    global = moduleTranslation.lookupGlobal(variable);
    assert(global && "could not find the translated host symbol");
    auto *llvmVariable = cast<llvm::GlobalVariable>(global);
    size = m->getDataLayout().getTypeStoreSize(llvmVariable->getValueType());
  } else {
    return op->emitError("unsupported operation for '")
           << acc::getOffloadEntryAttrName() << "'";
  }

  StringRef symbolName = global->getName();
  llvm::offloading::emitOffloadingEntry(*m, llvm::object::OFK_OpenACC, global,
                                        symbolName, size,
                                        /*Flags=*/0, /*Data=*/0);
  return success();
}

namespace {
class OpenACCDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  amendOperation(Operation *op, ArrayRef<llvm::Instruction *> instructions,
                 NamedAttribute attribute,
                 LLVM::ModuleTranslation &moduleTranslation) const override {
    if (attribute.getName() == acc::getOffloadEntryAttrName())
      return convertOffloadEntry(op, moduleTranslation);
    return success();
  }
};
} // namespace

void mlir::registerOpenACCDialectTranslation(DialectRegistry &registry) {
  registry.insert<acc::OpenACCDialect>();
  registry.addExtension(+[](MLIRContext *ctx, acc::OpenACCDialect *dialect) {
    dialect->addInterfaces<OpenACCDialectLLVMIRTranslationInterface>();
    acc::OffloadPackagerAttr::attachInterface<OffloadPackagerAttrImpl>(*ctx);
  });
}

void mlir::registerOpenACCDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerOpenACCDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
