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
// GPU kernel metadata attribute
//===----------------------------------------------------------------------===//

def GPU_KernelMetadataAttr : GPU_Attr<"KernelMetadata", "kernel_metadata"> {
  let description = [{
    GPU attribute for storing metadata related to a compiled kernel. The
    attribute contains the name and arguments type of the kernel.

    The attribute also contains optional parameters for storing the arguments
    attributes as well as a dictionary for additional metadata, like occupancy
    information or other function attributes.

    Note: The `arg_attrs` parameter is expected to follow all the constraints
    imposed by the `mlir::FunctionOpInterface` interface.

    Examples:
    ```mlir
      #gpu.kernel_metadata<@kernel1, (i32) -> (), arg_attrs = [...],  metadata = {reg_count = 255, ...}>
      #gpu.kernel_metadata<@kernel2, (i32, f64) -> ()>
    ```
  }];
  let parameters = (ins
    "StringAttr":$name,
    "Type":$function_type,
    OptionalParameter<"ArrayAttr", "arguments attributes">:$arg_attrs,
    OptionalParameter<"DictionaryAttr", "metadata dictionary">:$metadata
  );
  let assemblyFormat = [{
    `<` $name `,` $function_type (`,` struct($arg_attrs, $metadata)^)? `>`
  }];
  let builders = [
    AttrBuilderWithInferredContext<(ins "StringAttr":$name,
                                        "Type":$functionType,
                                        CArg<"ArrayAttr", "nullptr">:$argAttrs,
                                        CArg<"DictionaryAttr",
                                             "nullptr">:$metadata), [{
      assert(name && "invalid name");
      return $_get(name.getContext(), name, functionType, argAttrs, metadata);
    }]>,
    AttrBuilderWithInferredContext<(ins "FunctionOpInterface":$kernel,
                                         CArg<"DictionaryAttr",
                                              "nullptr">:$metadata)>
  ];
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    /// Compare two kernels based on the name.
    bool operator<(const KernelMetadataAttr& other) const {
      return getName().getValue() < other.getName().getValue();
    }

    /// Returns the metadata attribute corresponding to `key` or `nullptr`
    /// if missing.
    Attribute getAttr(StringRef key) const {
      DictionaryAttr attrs = getMetadata();
      return attrs ? attrs.get(key) : nullptr;
    }
    template <typename ConcreteAttr>
    ConcreteAttr getAttr(StringRef key) const {
      return llvm::dyn_cast_or_null<ConcreteAttr>(getAttr(key));
    }
    Attribute getAttr(StringAttr key) const {
      DictionaryAttr attrs = getMetadata();
      return attrs ? attrs.get(key) : nullptr;
    }
    template <typename ConcreteAttr>
    ConcreteAttr getAttr(StringAttr key) const {
      return llvm::dyn_cast_or_null<ConcreteAttr>(getAttr(key));
    }

    /// Returns the attribute dictionary at position `index`.
    DictionaryAttr getArgAttrDict(unsigned index) {
      ArrayAttr argArray = getArgAttrs();
      return argArray ? llvm::cast<DictionaryAttr>(argArray[index]) : nullptr;
    }

    /// Return the specified attribute, if present, for the argument at 'index',
    /// null otherwise.
    Attribute getArgAttr(unsigned index, StringAttr name) {
      DictionaryAttr argDict = getArgAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    Attribute getArgAttr(unsigned index, StringRef name) {
      DictionaryAttr argDict = getArgAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    /// Returns a new KernelMetadataAttr that contains `attrs` in the metadata dictionary.
    KernelMetadataAttr appendMetadata(ArrayRef<NamedAttribute> attrs) const;
  }];
}

//===----------------------------------------------------------------------===//
// GPU kernel table attribute
//===----------------------------------------------------------------------===//

def GPU_KernelTableAttr : GPU_Attr<"KernelTable", "kernel_table"> {
  let description = [{
    GPU attribute representing a list of `#gpu.kernel_metadata` attributes. This
    attribute supports searching kernels by name. All kernels in the table must
    have an unique name.

    Examples:
    ```mlir
      // Empty table.
      #gpu.kernel_table<>

      // Table with a single kernel.
      #gpu.kernel_table<[#gpu.kernel_metadata<kernel0, () -> () >]>

      // Table with multiple kernels.
      #gpu.kernel_table<[
        #gpu.kernel_metadata<"kernel0", (i32, f32) -> (), metadata = {sgpr_count = 255}>,
        #gpu.kernel_metadata<"kernel1", (i32) -> ()>
      ]>
    ```
  }];
  let parameters = (ins
    OptionalArrayRefParameter<"KernelMetadataAttr", "array of kernels">:$kernel_table
  );
  let assemblyFormat = [{
    `<` (`[` qualified($kernel_table)^ `]`)? `>`
  }];
  let builders = [
    AttrBuilder<(ins "ArrayRef<KernelMetadataAttr>":$kernels,
                     CArg<"bool", "false">:$isSorted)>
  ];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    llvm::ArrayRef<KernelMetadataAttr>::iterator begin() const {
      return getKernelTable().begin();
    }
    llvm::ArrayRef<KernelMetadataAttr>::iterator end() const {
      return getKernelTable().end();
    }
    size_t size() const {
      return getKernelTable().size();
    }
    bool empty() const {
      return getKernelTable().empty();
    }

    /// Returns the kernel with name `key` or `nullptr` if not present.
    KernelMetadataAttr lookup(StringRef key) const;
    KernelMetadataAttr lookup(StringAttr key) const;
  }];
}

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
    binary string with the object, the object properties, and kernel metadata,
    encapsulating how the object was generated and its properties with the
    object itself.

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

    Kernel metadata is specified through the `kernels` parameter, and can be
    used to specify additional information on a kernel by kernel basis.

    The target attribute must implement or promise the `TargetAttrInterface`
    interface.

    ```
      #gpu.object<#rocdl.target, offload = "..."> // An offload object.
      #gpu.object<#nvvm.target, properties = {O = 3 : i32}, assembly = "..."> // An assembly object with additional properties.
      #gpu.object<#rocdl.target, bin = "..."> // A binary object.
      #gpu.object<#nvvm.target, "..."> // A fatbin object.
      #gpu.object<#nvvm.target, kernels = #gpu.kernel_table<...>, "..."> // An object with a kernel table.
    ```
  }];
  let parameters = (ins
    "Attribute":$target,
    DefaultValuedParameter<"CompilationTarget", "CompilationTarget::Fatbin">:$format,
    "StringAttr":$object,
    OptionalParameter<"DictionaryAttr">:$properties,
    OptionalParameter<"KernelTableAttr">:$kernels
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins "Attribute":$target,
                                        "CompilationTarget":$format,
                                        "StringAttr":$object,
                                        CArg<"DictionaryAttr", "nullptr">:$properties,
                                        CArg<"KernelTableAttr", "nullptr">:$kernels), [{
      assert(target && "invalid target");
      return $_get(target.getContext(), target, format, object, properties, kernels);
    }]>
  ];
  let assemblyFormat = [{ `<`
      $target `,`  (`properties` `=` $properties^ `,`)?
      (`kernels` `=` $kernels^ `,`)?
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
