//===-- CompilationAttrInterfaces.td - GPU compilation interfaces ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces for GPU compilation attributes.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_COMPILATIONATTRINTERFACES
#define GPU_COMPILATIONATTRINTERFACES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// GPU target attribute interface.
//===----------------------------------------------------------------------===//
def GPUTargetAttrInterface : AttrInterface<"TargetAttrInterface"> {
  let description = [{
    Interface for GPU target attributes. Attributes implementing this interface
    compile GPU modules into binary objects, providing an opaque interface to
    hide implementation details.
  }];
  let cppNamespace = "::mlir::gpu";
  let methods = [
    InterfaceMethod<[{
        Serializes a GPU module to a string containing a representation of the
        module.

        If serialization fails then the method should return `std::nullopt`.

        The `module` argument must be a GPU Module Op. The `options` argument is
        meant to be used for passing additional options that are not in the
        attribute.
      }],
      "std::optional<SmallVector<char, 0>>", "serializeToObject",
      (ins "Operation*":$module, "const gpu::TargetOptions&":$options)>
  ];
}

def GPUTargetAttr :
    ConfinedAttr<AnyAttr, [PromisedAttrInterface<GPUTargetAttrInterface>]> {
  let description = [{
    Generic GPU target attribute. These attributes must implement or promise
    the `GPUTargetAttrInterface` interface.
  }];
}

def GPUTargetArrayAttr : TypedArrayAttrBase<GPUTargetAttr,
  "array of GPU target attributes">;

def GPUNonEmptyTargetArrayAttr :
  ConfinedAttr<GPUTargetArrayAttr, [ArrayMinCount<1>]>;

//===----------------------------------------------------------------------===//
// GPU offloading translation attribute trait.
//===----------------------------------------------------------------------===//

def OffloadingTranslationAttrTrait :
   NativeTrait<"OffloadingTranslationAttrTrait", ""> {
  let cppNamespace = "::mlir::gpu";
}

def HasOffloadingTranslationAttrTrait : AttrConstraint<
  CPred<"$_self.hasTrait<::mlir::gpu::OffloadingTranslationAttrTrait>()">,
  "with the `OffloadingTranslationAttrTrait` trait."
>;

def OffloadingTranslationAttr :
    ConfinedAttr<AnyAttr, [HasOffloadingTranslationAttrTrait]> {
  let description = [{
    Generic GPU offloading translation attribute. These attributes must
    implement an interface for handling the translation of GPU offloading
    operations like `gpu.binary` & `gpu.launch_func`. An example of such
    interface is the `OffloadingLLVMTranslationAttrInterface` interface.
  }];
}

//===----------------------------------------------------------------------===//
// GPU offloading LLVM translation handler attribute interface.
//===----------------------------------------------------------------------===//

def OffloadingLLVMTranslationAttrInterface :
    AttrInterface<"OffloadingLLVMTranslationAttrInterface"> {
  let description = [{
    Interface for GPU offloading LLVM translation attributes. Attributes
    implementing this interface manage the interaction between GPU offloading
    operations and host IR.
  }];
  let cppNamespace = "::mlir::gpu";
  let methods = [
    InterfaceMethod<[{
        Translates a `gpu.binary` Op into a sequence of LLVM IR target-specific
        instructions, embedding the binary into a host LLVM module.

        The LLVM translation mechanism invokes this function when translating a
        `gpu.binary`.

        The first argument has to be a GPU binary operation.
        If the function fails at any point, it must return `failure`.
      }],
      "LogicalResult", "embedBinary",
      (ins "Operation*":$binaryOp, "llvm::IRBuilderBase&":$hostBuilder,
           "LLVM::ModuleTranslation&":$hostModuleTranslation)
    >,
    InterfaceMethod<[{
        Translates a `gpu.launch_func` op into a sequence of LLVM IR
        target-specific instructions, resulting in a kernel launch on host IR.

        The LLVM translation mechanism invokes this function when translating a
        `gpu.launch_func` operation; it searches the appropriate binary and uses
        its offloading handler.

        The first two arguments must be GPU launch and binary operations,
        respectively. If the function fails at any point, it must return
        `failure`.
      }],
      "LogicalResult", "launchKernel",
      (ins "Operation*":$launchFunc, "Operation*":$binaryOp,
           "llvm::IRBuilderBase&":$hostBuilder,
           "LLVM::ModuleTranslation&":$hostModuleTranslation)
    >
  ];
}

#endif // GPU_COMPILATIONATTRINTERFACES
