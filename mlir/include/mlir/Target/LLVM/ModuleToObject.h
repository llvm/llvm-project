//===- ModuleToObject.h - Module to object base class -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the base class for transforming operations into binary
// objects.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVM_MODULETOOBJECT_H
#define MLIR_TARGET_LLVM_MODULETOOBJECT_H

#include "mlir/IR/Operation.h"
#include "llvm/IR/Module.h"

namespace llvm {
class TargetMachine;
} // namespace llvm

namespace mlir {
namespace LLVM {
class ModuleTranslation;
/// Utility base class for transforming operations into binary objects, by
/// default it returns the serialized LLVM bitcode for the module. The
/// operations being transformed must be translatable into LLVM IR.
class ModuleToObject {
public:
  ModuleToObject(Operation &module, StringRef triple, StringRef chip,
                 StringRef features = {}, int optLevel = 3);
  virtual ~ModuleToObject() = default;

  /// Returns the operation being serialized.
  Operation &getOperation();

  /// Runs the serialization pipeline, returning `std::nullopt` on error.
  virtual std::optional<SmallVector<char, 0>> run();

protected:
  // Hooks to be implemented by derived classes.

  /// Hook for loading bitcode files, returns std::nullopt on failure.
  virtual std::optional<SmallVector<std::unique_ptr<llvm::Module>>>
  loadBitcodeFiles(llvm::Module &module, llvm::TargetMachine &targetMachine) {
    return SmallVector<std::unique_ptr<llvm::Module>>();
  }

  /// Hook for performing additional actions on a loaded bitcode file.
  virtual LogicalResult handleBitcodeFile(llvm::Module &module,
                                          llvm::TargetMachine &targetMachine) {
    return success();
  }

  /// Hook for performing additional actions on the llvmModule pre linking.
  virtual void handleModulePreLink(llvm::Module &module,
                                   llvm::TargetMachine &targetMachine) {}

  /// Hook for performing additional actions on the llvmModule post linking.
  virtual void handleModulePostLink(llvm::Module &module,
                                    llvm::TargetMachine &targetMachine) {}

  /// Serializes the LLVM IR bitcode to an object file, by default it serializes
  /// to LLVM bitcode.
  virtual std::optional<SmallVector<char, 0>>
  moduleToObject(llvm::Module &llvmModule, llvm::TargetMachine &targetMachine);

protected:
  /// Create the target machine based on the target triple and chip.
  std::unique_ptr<llvm::TargetMachine> createTargetMachine();

  /// Loads a bitcode file from path.
  std::unique_ptr<llvm::Module>
  loadBitcodeFile(llvm::LLVMContext &context,
                  llvm::TargetMachine &targetMachine, StringRef path);

  /// Loads multiple bitcode files.
  LogicalResult loadBitcodeFilesFromList(
      llvm::LLVMContext &context, llvm::TargetMachine &targetMachine,
      ArrayRef<std::string> fileList,
      SmallVector<std::unique_ptr<llvm::Module>> &llvmModules,
      bool failureOnError = true);

  /// Translates the operation to LLVM IR.
  std::unique_ptr<llvm::Module>
  translateToLLVMIR(llvm::LLVMContext &llvmContext);

  /// Link the llvmModule to other bitcode file.
  LogicalResult linkFiles(llvm::Module &module,
                          SmallVector<std::unique_ptr<llvm::Module>> &&libs);

  /// Optimize the module.
  LogicalResult optimizeModule(llvm::Module &module,
                               llvm::TargetMachine &targetMachine, int optL);

  /// Utility function for translating to ISA, returns `std::nullopt` on
  /// failure.
  static std::optional<std::string>
  translateToISA(llvm::Module &llvmModule, llvm::TargetMachine &targetMachine);

protected:
  /// Module to transform to a binary object.
  Operation &module;

  /// Target triple.
  StringRef triple;

  /// Target chip.
  StringRef chip;

  /// Target features.
  StringRef features;

  /// Optimization level.
  int optLevel;
};
} // namespace LLVM
} // namespace mlir

#endif // MLIR_TARGET_LLVM_MODULETOOBJECT_H
