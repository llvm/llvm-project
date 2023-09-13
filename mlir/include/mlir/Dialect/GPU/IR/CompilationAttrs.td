//===-- CompilationAttrs.td - GPU compilation attributes ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines GPU compilation related attributes.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_COMPILATION_ATTRS
#define GPU_COMPILATION_ATTRS

include "mlir/Dialect/GPU/IR/GPUBase.td"
include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"

//===----------------------------------------------------------------------===//
// GPU object attribute.
//===----------------------------------------------------------------------===//

def GPU_ObjectAttr : GPU_Attr<"Object", "object"> {
  let description = [{
    A GPU object attribute pairs a GPU target with a binary string,
    encapsulating the information of how the object was generated with the
    object itself.

    The target attribute must implement the `TargetAttrInterface` interface.

    ```
      #gpu.object<#nvvm.target, "...">
    ```
  }];
  let parameters = (ins "Attribute":$target, "StringAttr":$object);
  let assemblyFormat = [{`<` $target `,` $object `>`}];
  let genVerifyDecl = 1;
}

def GPUObjectArrayAttr :
  TypedArrayAttrBase<GPU_ObjectAttr, "an array of GPU object attributes">;

//===----------------------------------------------------------------------===//
// GPU offloading LLVM translation handler attributes.
//===----------------------------------------------------------------------===//

def GPU_SelectObjectAttr : GPU_Attr<"SelectObject", "select_object", [
      OffloadingTranslationAttrTrait
    ]> {
  let description = [{
    This GPU offloading handler selects a single GPU object for embedding. The
    object is selected based on the `target` parameter, this parameter can be
    either a number -i.e. selects the ith-target, or the target itself -i.e.
    searches for the specified target in the object array.

    The first object in a `gpu.binary` operation is selected if no target is
    specified.
  }];
  let parameters = (ins
    OptionalParameter<"Attribute", "Target to select for embedding.">:$target
  );
  let assemblyFormat = [{
    (`<` $target^ `>`)?
  }];
  let genVerifyDecl = 1;
}

#endif // GPU_COMPILATION_ATTRS
