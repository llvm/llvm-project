//===- DataLayoutInterfaces.td - Data layout interfaces ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interfaces for the data layout specification, operations to which
// they can be attached, and types that are subject to data layout.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DATALAYOUTINTERFACES
#define MLIR_DATALAYOUTINTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Attribute interfaces
//===----------------------------------------------------------------------===//

def DataLayoutEntryInterface : AttrInterface<"DataLayoutEntryInterface"> {
  let cppNamespace = "::mlir";

  let description = [{
    Attribute interface describing an entry in a data layout specification.

    A data layout specification entry is a key-value pair. Its key is either a
    type, when the entry is related to a type or a class of types, or an
    identifier, when it is not. `DataLayoutEntryKey` is an alias allowing one
    to use both key types. Its value is an arbitrary attribute that is
    interpreted either by the type for type keys or by the dialect containing
    the identifier for identifier keys. The interface provides a hook that
    can be used by specific implementations to delegate the verification of
    attribute fitness for a particular key to the relevant type or dialect.
  }];

  let methods = [
    InterfaceMethod<
      /*description=*/"Returns the key of the this layout entry.",
      /*retTy=*/"::mlir::DataLayoutEntryKey",
      /*methodName=*/"getKey",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*description=*/"Returns the value of this layout entry.",
      /*retTy=*/"::mlir::Attribute",
      /*methodName=*/"getValue",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*description=*/"Checks that the entry is well-formed, reports errors "
                      "at the provided location.",
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"verifyEntry",
      /*args=*/(ins "::mlir::Location":$loc),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return ::mlir::success(); }]
    >
  ];

  let extraClassDeclaration = [{
    /// Returns `true` if the key of this entry is a type.
    bool isTypeEntry() {
      return getKey().is<::mlir::Type>();
    }
  }];
}

def DataLayoutSpecInterface : AttrInterface<"DataLayoutSpecInterface"> {
  let cppNamespace = "::mlir";

  let description = [{
    Attribute interface describing a data layout specification.

    A data layout specification is seen as a sequence of entries, each of which
    is an attribute implementing the data layout entry interface. It assumes
    a contiguous underlying storage for entries. The interface provides a hook
    for implementations to verify the well-formedness of the specification,
    with a default implementation that verifies the absence of entries with
    duplicate keys and the well-formedness of each individual entry before
    dispatching to the type or dialect the entry is associated with.

    Data layout specifications may need to be combined in case they appear on
    nested operations subject to layout, or to ensure the validity of layout
    modification. Concrete specification attributes must implement the
    corresponding hook.
  }];
  // The underlying storage being contiguous may be revised in the future, but
  // care must be taken to avoid materializing or copying the entire list of
  // entries.

  let methods = [
    InterfaceMethod<
      /*description=*/"Combines the current layout with the given list of "
                      "layouts, provided from the outermost (oldest) to the "
                      "innermost (newest). Returns null on failure.",
      /*retTy=*/"::mlir::DataLayoutSpecInterface",
      /*methodName=*/"combineWith",
      /*args=*/(ins "::llvm::ArrayRef<::mlir::DataLayoutSpecInterface>":$specs)
    >,
    InterfaceMethod<
      /*description=*/"Returns the list of layout entries.",
      /*retTy=*/"::mlir::DataLayoutEntryListRef",
      /*methodName=*/"getEntries",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*description=*/"Returns the alloca memory space identifier.",
      /*retTy=*/"::mlir::StringAttr",
      /*methodName=*/"getAllocaMemorySpaceIdentifier",
      /*args=*/(ins "::mlir::MLIRContext *":$context)
    >,
    InterfaceMethod<
      /*description=*/"Returns the program memory space identifier.",
      /*retTy=*/"::mlir::StringAttr",
      /*methodName=*/"getProgramMemorySpaceIdentifier",
      /*args=*/(ins "::mlir::MLIRContext *":$context)
    >,
    InterfaceMethod<
      /*description=*/"Returns the global memory space identifier.",
      /*retTy=*/"::mlir::StringAttr",
      /*methodName=*/"getGlobalMemorySpaceIdentifier",
      /*args=*/(ins "::mlir::MLIRContext *":$context)
    >,
    InterfaceMethod<
      /*description=*/"Returns the stack alignment identifier.",
      /*retTy=*/"::mlir::StringAttr",
      /*methodName=*/"getStackAlignmentIdentifier",
      /*args=*/(ins "::mlir::MLIRContext *":$context)
    >,
    // Implementations may override this if they have an efficient lookup
    // mechanism.
    InterfaceMethod<
      /*description=*/"Returns a copy of the entries related to a specific "
                      "type class regardles of type parameters. ",
      /*retTy=*/"::mlir::DataLayoutEntryList",
      /*methodName=*/"getSpecForType",
      /*args=*/(ins "::mlir::TypeID":$type),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::detail::filterEntriesForType($_attr.getEntries(), type);
      }]
    >,
    // Implementations may override this if they have an efficient lookup
    // mechanism.
    InterfaceMethod<
      /*description=*/"Returns the entry related to the given identifier, if "
                      "present.",
      /*retTy=*/"::mlir::DataLayoutEntryInterface",
      /*methodName=*/"getSpecForIdentifier",
      /*args=*/(ins "::mlir::StringAttr":$identifier),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::detail::filterEntryForIdentifier($_attr.getEntries(),
                                                        identifier);
      }]
    >,
    InterfaceMethod<
      /*description=*/"Verifies the validity of the specification and reports "
                      "any errors at the given location.",
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"verifySpec",
      /*args=*/(ins "::mlir::Location":$loc),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::detail::verifyDataLayoutSpec($_attr, loc);
      }]
    >,
  ];

  let extraClassDeclaration = [{
    /// Returns a copy of the entries related to the type specified as template
    /// parameter.
    template <typename Ty>
    DataLayoutEntryList getSpecForType() {
      return getSpecForType(TypeID::get<Ty>());
    }

    /// Populates the given maps with lists of entries grouped by the type or
    /// identifier they are associated with. Users are not expected to call this
    /// method directly.
    void bucketEntriesByType(
        ::llvm::DenseMap<::mlir::TypeID, ::mlir::DataLayoutEntryList> &types,
        ::llvm::DenseMap<::mlir::StringAttr,
                         ::mlir::DataLayoutEntryInterface> &ids);
  }];
}

//===----------------------------------------------------------------------===//
// Operation interface
//===----------------------------------------------------------------------===//

def DataLayoutOpInterface : OpInterface<"DataLayoutOpInterface"> {
  let cppNamespace = "::mlir";

  let description = [{
    Interface for operations that can have a data layout specification attached.

    The `DataLayout` object, which can be used for data layout queries, can be
    constructed for such operations. The absence of a data layout specification
    must be handled without failing.

    Concrete operations must implement the hook returning the data layout
    specification. They may optionally override the methods used in data layout
    queries, default implementations of which provide predefined answers for
    built-in types and dispatch to the type interface for all other types. These
    methods must be idempotent, that is return the same result on repeated
    queries with the same parameters. They are declared static and therefore
    have no access to the operation or its attributes. Instead, they receive a
    list of data layout entries relevant to the request. The entries are known
    to have passed the spec and entry verifier.
  }];

  let methods = [
    InterfaceMethod<
      /*description=*/"Returns the data layout specification for this op, or "
                      "null if it does not exist.",
      /*retTy=*/"::mlir::DataLayoutSpecInterface",
      /*methodName=*/"getDataLayoutSpec",
      /*args=*/(ins)
    >,
    StaticInterfaceMethod<
      /*description=*/"Returns the size of the given type computed using the "
                      "relevant entries. The data layout object can be used "
                      "for recursive queries.",
      /*retTy=*/"::llvm::TypeSize",
      /*methodName=*/"getTypeSize",
      /*args=*/(ins "::mlir::Type":$type,
                    "const ::mlir::DataLayout &":$dataLayout,
                    "::mlir::DataLayoutEntryListRef":$params),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ::llvm::TypeSize bits = ConcreteOp::getTypeSizeInBits(type, dataLayout, params);
        return ::mlir::detail::divideCeil(bits, 8u);
      }]
    >,
    StaticInterfaceMethod<
      /*description=*/"Returns the size of the given type in bits computed "
                      "using the relevant entries. The data layout object can "
                      "be used for recursive queries.",
      /*retTy=*/"::llvm::TypeSize",
      /*methodName=*/"getTypeSizeInBits",
      /*args=*/(ins "::mlir::Type":$type,
                    "const ::mlir::DataLayout &":$dataLayout,
                    "::mlir::DataLayoutEntryListRef":$params),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::detail::getDefaultTypeSizeInBits(type, dataLayout,
                                                        params);
      }]
    >,
    StaticInterfaceMethod<
      /*description=*/"Returns the alignment required by the ABI for the given "
                      "type computed using the relevant entries. The data "
                      "layout object can be used for recursive queries.",
      /*retTy=*/"uint64_t",
      /*methodName=*/"getTypeABIAlignment",
      /*args=*/(ins "::mlir::Type":$type,
                    "const ::mlir::DataLayout &":$dataLayout,
                    "::mlir::DataLayoutEntryListRef":$params),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::detail::getDefaultABIAlignment(type, dataLayout, params);
      }]
    >,
    StaticInterfaceMethod<
      /*description=*/"Returns the alignment preferred by the given type "
                      "computed using the relevant entries. The data layout"
                      "object can be used for recursive queries.",
      /*retTy=*/"uint64_t",
      /*methodName=*/"getTypePreferredAlignment",
      /*args=*/(ins "::mlir::Type":$type,
                    "const ::mlir::DataLayout &":$dataLayout,
                    "::mlir::DataLayoutEntryListRef":$params),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::detail::getDefaultPreferredAlignment(type, dataLayout,
                                                            params);
      }]
    >,
    StaticInterfaceMethod<
      /*description=*/"Returns the bitwidth that should be used when "
                      "performing index computations for the type computed "
                      "using the relevant entries. The data layout object can "
                      "be used for recursive queries.",
      /*retTy=*/"std::optional<uint64_t>",
      /*methodName=*/"getIndexBitwidth",
      /*args=*/(ins "::mlir::Type":$type,
                    "const ::mlir::DataLayout &":$dataLayout,
                    "::mlir::DataLayoutEntryListRef":$params),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::detail::getDefaultIndexBitwidth(type, dataLayout,
                                                       params);
      }]
    >,
    StaticInterfaceMethod<
      /*description=*/"Returns the memory space used by the ABI computed "
                      "using the relevant entries. The data layout object "
                      "can be used for recursive queries.",
      /*retTy=*/"::mlir::Attribute",
      /*methodName=*/"getAllocaMemorySpace",
      /*args=*/(ins "::mlir::DataLayoutEntryInterface":$entry),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::detail::getDefaultAllocaMemorySpace(entry);
      }]
    >,
    StaticInterfaceMethod<
      /*description=*/"Returns the memory space used by the ABI computed "
                      "using the relevant entries. The data layout object "
                      "can be used for recursive queries.",
      /*retTy=*/"::mlir::Attribute",
      /*methodName=*/"getProgramMemorySpace",
      /*args=*/(ins "::mlir::DataLayoutEntryInterface":$entry),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::detail::getDefaultProgramMemorySpace(entry);
      }]
    >,
    StaticInterfaceMethod<
      /*description=*/"Returns the memory space used by the ABI computed "
                      "using the relevant entries. The data layout object "
                      "can be used for recursive queries.",
      /*retTy=*/"::mlir::Attribute",
      /*methodName=*/"getGlobalMemorySpace",
      /*args=*/(ins "::mlir::DataLayoutEntryInterface":$entry),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::detail::getDefaultGlobalMemorySpace(entry);
      }]
    >,
    StaticInterfaceMethod<
      /*description=*/"Returns the natural stack alignment in bits computed "
                      "using the relevant entries. The data layout object "
                      "can be used for recursive queries.",
      /*retTy=*/"uint64_t",
      /*methodName=*/"getStackAlignment",
      /*args=*/(ins "::mlir::DataLayoutEntryInterface":$entry),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::detail::getDefaultStackAlignment(entry);
      }]
    >,
  ];

  let verify = [{ return ::mlir::detail::verifyDataLayoutOp($_op); }];
}

//===----------------------------------------------------------------------===//
// Type interface
//===----------------------------------------------------------------------===//

def DataLayoutTypeInterface : TypeInterface<"DataLayoutTypeInterface"> {
  let cppNamespace = "::mlir";

  let description = [{
    Interface for types subject to data layout.

    Types willing to be supported by the data layout subsystem should implement
    this interface by providing implementations of functions querying their
    size, required and preferred alignment. Each of these functions accepts as
    arguments a data layout object that can be used to perform recursive queries
    in the same scope, and a list of data layout entries relevant to this type.
    Specifically, the entries are those that have as key _any instance_ of the
    same type class as the current type. For example, if IntegerType had
    implemented this interface, it would have received the entries with keys i1,
    i2, i8, etc. regardless of the bitwidth of this type. This mechanism allows
    types to "interpolate" the results in a type-specific way instead of listing
    all possible types in the specification.

    The list of entries may be empty, in which case the type must provide a
    reasonable default value. The entries in the list are known to have passed
    the spec and the entry verifiers, as well as the type-specified verifier if
    provided.

    In case of nested layout specs or spec changes, the type can override a hook
    indicating whether the outer (old) and the inner (new) spec are compatible.
  }];

  let methods = [
    InterfaceMethod<
      /*description=*/"Returns the size of this type in bytes.",
      /*retTy=*/"::llvm::TypeSize",
      /*methodName=*/"getTypeSize",
      /*args=*/(ins "const ::mlir::DataLayout &":$dataLayout,
                    "::mlir::DataLayoutEntryListRef":$params),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ::llvm::TypeSize bits = $_type.getTypeSizeInBits(dataLayout, params);
        return ::mlir::detail::divideCeil(bits, 8u);
      }]
    >,
    InterfaceMethod<
      /*description=*/"Returns the size of this type in bits.",
      /*retTy=*/"::llvm::TypeSize",
      /*methodName=*/"getTypeSizeInBits",
      /*args=*/(ins "const ::mlir::DataLayout &":$dataLayout,
                    "::mlir::DataLayoutEntryListRef":$params)
    >,
    InterfaceMethod<
      /*description=*/"Returns the ABI-required alignment for this type, "
                      "in bytes",
      /*retTy=*/"uint64_t",
      /*methodName=*/"getABIAlignment",
      /*args=*/(ins "const ::mlir::DataLayout &":$dataLayout,
                    "::mlir::DataLayoutEntryListRef":$params)
    >,
    InterfaceMethod<
      /*description=*/"Returns the preferred alignment for this type, "
                      "in bytes.",
      /*retTy=*/"uint64_t",
      /*methodName=*/"getPreferredAlignment",
      /*args=*/(ins "const ::mlir::DataLayout &":$dataLayout,
                    "::mlir::DataLayoutEntryListRef":$params)
    >,
    InterfaceMethod<
      /*description=*/"Returns the bitwidth that should be used when "
                      "performing index computations for the given "
                      "pointer-like type. If the type is not a pointer-like "
                      "type, returns std::nullopt.",
      /*retTy=*/"std::optional<uint64_t>",
      /*methodName=*/"getIndexBitwidth",
      /*args=*/(ins "const ::mlir::DataLayout &":$dataLayout,
                    "::mlir::DataLayoutEntryListRef":$params),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return std::nullopt; }]
    >,
    InterfaceMethod<
      /*desc=*/"Returns true if the two lists of entries are compatible, that "
               "is, that `newLayout` spec entries can be nested in an op with "
               "`oldLayout` spec entries.",
      /*retTy=*/"bool",
      /*methodName=*/"areCompatible",
      /*args=*/(ins "::mlir::DataLayoutEntryListRef":$oldLayout,
                    "::mlir::DataLayoutEntryListRef":$newLayout),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return true; }]
    >,
    InterfaceMethod<
      /*description=*/"Verifies that the given list of entries is valid for "
                      "this type.",
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"verifyEntries",
      /*args=*/(ins "::mlir::DataLayoutEntryListRef":$entries,
                    "::mlir::Location":$loc),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return ::mlir::success(); }]
    >,
  ];
}

#endif // MLIR_DATALAYOUTINTERFACES
