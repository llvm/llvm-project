//===-- CompilationAttrInterfaces.td - GPU compilation interfaces ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces for GPU target attributes.
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

def GPUTargetArrayAttr : TypedArrayAttrBase<GPUTargetAttrInterface,
  "array of GPU target attributes">;

def GPUNonEmptyTargetArrayAttr :
  ConfinedAttr<GPUTargetArrayAttr, [ArrayMinCount<1>]>;

#endif // GPU_COMPILATIONATTRINTERFACES
