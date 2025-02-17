//===-- CompilationInterfaces.h - GPU compilation interfaces  ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces for GPU compilation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_IR_COMPILATIONINTERFACES_H
#define MLIR_DIALECT_GPU_IR_COMPILATIONINTERFACES_H

#include "mlir/IR/Attributes.h"
#include "llvm/IR/Module.h"

namespace llvm {
class IRBuilderBase;
}

namespace mlir {
class SymbolTable;
namespace LLVM {
class ModuleTranslation;
}
namespace gpu {
enum class CompilationTarget : uint32_t;
constexpr StringLiteral elfSectionName = "section";

/// This class indicates that the attribute associated with this trait is a GPU
/// offloading translation attribute. These kinds of attributes must implement
/// an interface for handling the translation of GPU offloading operations like
/// `gpu.binary` & `gpu.launch_func`.
template <typename ConcreteType>
class OffloadingTranslationAttrTrait
    : public AttributeTrait::TraitBase<ConcreteType,
                                       OffloadingTranslationAttrTrait> {
  // TODO: Verify the attribute promises or implements the interface.
};

/// This class serves as an opaque interface for passing options to the
/// `TargetAttrInterface` methods. Users of this class must implement the
/// `classof` method as well as using the macros `MLIR_*_EXPLICIT_TYPE_ID` to
/// ensure type safeness. Targets are free to ignore these options.
class TargetOptions {
public:
  /// Constructor initializing the toolkit path, the list of files to link to,
  /// extra command line options, the compilation target and a callback for
  /// obtaining the parent symbol table. The default compilation target is
  /// `Fatbin`.
  TargetOptions(
      StringRef toolkitPath = {}, ArrayRef<Attribute> librariesToLink = {},
      StringRef cmdOptions = {}, StringRef elfSection = {},
      CompilationTarget compilationTarget = getDefaultCompilationTarget(),
      function_ref<SymbolTable *()> getSymbolTableCallback = {},
      function_ref<void(llvm::Module &)> initialLlvmIRCallback = {},
      function_ref<void(llvm::Module &)> linkedLlvmIRCallback = {},
      function_ref<void(llvm::Module &)> optimizedLlvmIRCallback = {},
      function_ref<void(StringRef)> isaCallback = {});

  /// Returns the typeID.
  TypeID getTypeID() const;

  /// Returns the toolkit path.
  StringRef getToolkitPath() const;

  /// Returns the LLVM libraries to link to.
  ArrayRef<Attribute> getLibrariesToLink() const;

  /// Returns the command line options.
  StringRef getCmdOptions() const;

  /// Returns the ELF section.
  StringRef getELFSection() const;

  /// Returns a tokenization of the command line options.
  std::pair<llvm::BumpPtrAllocator, SmallVector<const char *>>
  tokenizeCmdOptions() const;

  /// Returns the compilation target.
  CompilationTarget getCompilationTarget() const;

  /// Returns the result of the `getSymbolTableCallback` callback or a nullptr
  /// if no callback was provided.
  /// Note: The callback itself can return nullptr. It is up to the target how
  /// to react to getting a nullptr, e.g., emitting an error or constructing the
  /// table.
  SymbolTable *getSymbolTable() const;

  /// Returns the callback invoked with the initial LLVM IR for the device
  /// module.
  function_ref<void(llvm::Module &)> getInitialLlvmIRCallback() const;

  /// Returns the callback invoked with LLVM IR for the device module
  /// after linking the device libraries.
  function_ref<void(llvm::Module &)> getLinkedLlvmIRCallback() const;

  /// Returns the callback invoked with LLVM IR for the device module after
  /// LLVM optimizations but before codegen.
  function_ref<void(llvm::Module &)> getOptimizedLlvmIRCallback() const;

  /// Returns the callback invoked with the target ISA for the device,
  /// for example PTX assembly.
  function_ref<void(StringRef)> getISACallback() const;

  /// Returns the default compilation target: `CompilationTarget::Fatbin`.
  static CompilationTarget getDefaultCompilationTarget();

  /// Returns a tokenization of the command line options.
  static std::pair<llvm::BumpPtrAllocator, SmallVector<const char *>>
  tokenizeCmdOptions(const std::string &cmdOptions);

protected:
  /// Derived classes must use this constructor to initialize `typeID` to the
  /// appropiate value: ie. `TargetOptions(TypeID::get<DerivedClass>())`.
  TargetOptions(
      TypeID typeID, StringRef toolkitPath = {},
      ArrayRef<Attribute> librariesToLink = {}, StringRef cmdOptions = {},
      StringRef elfSection = {},
      CompilationTarget compilationTarget = getDefaultCompilationTarget(),
      function_ref<SymbolTable *()> getSymbolTableCallback = {},
      function_ref<void(llvm::Module &)> initialLlvmIRCallback = {},
      function_ref<void(llvm::Module &)> linkedLlvmIRCallback = {},
      function_ref<void(llvm::Module &)> optimizedLlvmIRCallback = {},
      function_ref<void(StringRef)> isaCallback = {});

  /// Path to the target toolkit.
  std::string toolkitPath;

  /// List of files to link with the LLVM module.
  SmallVector<Attribute> librariesToLink;

  /// An optional set of command line options to be used by the compilation
  /// process.
  std::string cmdOptions;

  /// ELF Section where the binary needs to be located
  std::string elfSection;

  /// Compilation process target format.
  CompilationTarget compilationTarget;

  /// Callback for obtaining the parent symbol table of all the GPU modules
  /// being serialized.
  function_ref<SymbolTable *()> getSymbolTableCallback;

  /// Callback invoked with the initial LLVM IR for the device module.
  function_ref<void(llvm::Module &)> initialLlvmIRCallback;

  /// Callback invoked with LLVM IR for the device module after
  /// linking the device libraries.
  function_ref<void(llvm::Module &)> linkedLlvmIRCallback;

  /// Callback invoked with LLVM IR for the device module after
  /// LLVM optimizations but before codegen.
  function_ref<void(llvm::Module &)> optimizedLlvmIRCallback;

  /// Callback invoked with the target ISA for the device,
  /// for example PTX assembly.
  function_ref<void(StringRef)> isaCallback;

private:
  TypeID typeID;
};
} // namespace gpu
} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::TargetOptions)

#include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.h.inc"

#endif // MLIR_DIALECT_GPU_IR_COMPILATIONINTERFACES_H
