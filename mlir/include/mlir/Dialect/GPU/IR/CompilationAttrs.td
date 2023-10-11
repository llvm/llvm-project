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

// For documentation on this enum cases, see the `GPU_ObjectAttr` docs.
def GPU_ObjectOffload : I32EnumAttrCase<"Offload", 1, "offload">;
def GPU_ObjectISA : I32EnumAttrCase<"Assembly", 2, "assembly">;
def GPU_ObjectBinary : I32EnumAttrCase<"Binary", 3, "bin">;
def GPU_ObjectFatbin : I32EnumAttrCase<"Fatbin", 4, "fatbin">;
def GPU_CompilationTargetEnum : GPU_I32Enum<
  "CompilationTarget", "GPU compilation format", [
    GPU_ObjectOffload,
    GPU_ObjectISA,
    GPU_ObjectBinary,
    GPU_ObjectFatbin
  ]>;

def GPU_ObjectAttr : GPU_Attr<"Object", "object"> {
  let description = [{
    A GPU object attribute glues together a GPU target, the object kind, a
    binary string with the object, and the object properties, encapsulating how
    the object was generated and its properties with the object itself.

    There are four object formats:
    1. `Offload`: represents generic objects not described by the other three
    formats, and its meaning is target-dependent. For example, on the NVPTX and
    AMDGPU targets, this format is associated with LLVM bitcode.
    2. `Assembly`: represents GPU assembly code. For example, in the NVPTX
    target, assembly is PTX code, which can be JITted at runtime.
    3. `Binary`: represents executable code for a GPU single architecture. For
    example, PTX code that was compiled for a specific compute capability. Note
    that this format is likely to throw an error if there is an architecture
    mismatch between the compiled and running architecture.
    4. `Fatbin`: represents a GPU fat binary with executable code for multiple
    architectures. This format is the default; thus, it gets elided inassembly
    code.

    Object properties are specified through the `properties` dictionary
    attribute and can be used to define additional information.
    The target attribute must implement or promise the `TargetAttrInterface`
    interface.

    ```
      #gpu.object<#rocdl.target, offload = "..."> // An offload object.
      #gpu.object<#nvvm.target, properties = {O = 3 : i32}, assembly = "..."> // An assembly object with additional properties.
      #gpu.object<#rocdl.target, bin = "..."> // A binary object.
      #gpu.object<#nvvm.target, "..."> // A fatbin object.
    ```
  }];
  let parameters = (ins
    "Attribute":$target,
    DefaultValuedParameter<"CompilationTarget", "CompilationTarget::Fatbin">:$format,
    "StringAttr":$object,
    OptionalParameter<"DictionaryAttr">:$properties
  );
  let assemblyFormat = [{ `<`
      $target `,`  (`properties` `=` $properties ^ `,`)?
      custom<Object>($format, $object)
    `>`
  }];
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
