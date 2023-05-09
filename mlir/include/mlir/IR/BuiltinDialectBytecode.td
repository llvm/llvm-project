//===-- BuiltinBytecode.td - Builtin bytecode defs ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the Builtin bytecode reader/writer definition file.
//
//===----------------------------------------------------------------------===//

#ifndef BUILTIN_BYTECODE
#define BUILTIN_BYTECODE

include "mlir/IR/BytecodeBase.td"

def LocationAttr : AttributeKind;
def ShapedType: WithType<"ShapedType", Type>;

def Location : CompositeBytecode {
  dag members = (attr
    WithGetter<"(LocationAttr)$_attrType", WithType<"LocationAttr", LocationAttr>>:$value
  );
  let cBuilder = "Location($_args)";
}

def String :
  WithParser <"succeeded($_reader.readString($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeOwnedString($_getter)",
  WithGetter <"$_attrType",
  WithType   <"StringRef">>>>>;

// enum AttributeCode {
//  ///   ArrayAttr {
//  ///     elements: Attribute[]
//  ///   }
//  ///
//  kArrayAttr = 0,
//
def ArrayAttr : DialectAttribute<(attr
  Array<Attribute>:$value
)>;

let cType = "StringAttr" in {
//  ///   StringAttr {
//  ///     value: string
//  ///   }
//  kStringAttr = 2,
def StringAttr : DialectAttribute<(attr
  String:$value
)> {
  let printerPredicate = "::llvm::isa<NoneType>($_val.getType())";
}

//  ///   StringAttrWithType {
//  ///     value: string,
//  ///     type: Type
//  ///   }
//  /// A variant of StringAttr with a type.
//  kStringAttrWithType = 3,
def StringAttrWithType : DialectAttribute<(attr
  String:$value,
  Type:$type
)> { let printerPredicate = "!::llvm::isa<NoneType>($_val.getType())"; }
}

//  ///   DictionaryAttr {
//  ///     attrs: <StringAttr, Attribute>[]
//  ///   }
//  kDictionaryAttr = 1,
def NamedAttribute : CompositeBytecode {
  dag members = (attr
    StringAttr:$name,
    Attribute:$value
  );
  let cBuilder = "NamedAttribute($_args)";
}
def DictionaryAttr : DialectAttribute<(attr
  Array<NamedAttribute>:$value
)>;

//  ///   FlatSymbolRefAttr {
//  ///     rootReference: StringAttr
//  ///   }
//  /// A variant of SymbolRefAttr with no leaf references.
//  kFlatSymbolRefAttr = 4,
def FlatSymbolRefAttr: DialectAttribute<(attr
  StringAttr:$rootReference
)>;

//  ///   SymbolRefAttr {
//  ///     rootReference: StringAttr,
//  ///     leafReferences: FlatSymbolRefAttr[]
//  ///   }
//  kSymbolRefAttr = 5,
def SymbolRefAttr: DialectAttribute<(attr
  StringAttr:$rootReference,
  Array<FlatSymbolRefAttr>:$nestedReferences
)>;

//  ///   TypeAttr {
//  ///     value: Type
//  ///   }
//  kTypeAttr = 6,
def TypeAttr: DialectAttribute<(attr
  Type:$value
)>;

//  ///   UnitAttr {
//  ///   }
//  kUnitAttr = 7,
def UnitAttr: DialectAttribute<(attr)>;

//  ///   IntegerAttr {
//  ///     type: Type
//  ///     value: APInt,
//  ///   }
//  kIntegerAttr = 8,
def IntegerAttr: DialectAttribute<(attr
  Type:$type,
  KnownWidthAPInt<"type">:$value
)> {
  let cBuilder = "get<$_resultType>(context, type, *value)";
}

//
//  ///   FloatAttr {
//  ///     type: FloatType
//  ///     value: APFloat
//  ///   }
//  kFloatAttr = 9,
defvar FloatType = Type;
def FloatAttr : DialectAttribute<(attr
  FloatType:$type,
  KnownSemanticsAPFloat<"type">:$value
)> {
  let cBuilder = "get<$_resultType>(context, type, *value)";
}

//  ///   CallSiteLoc {
//  ///    callee: LocationAttr,
//  ///    caller: LocationAttr
//  ///   }
//  kCallSiteLoc = 10,
def CallSiteLoc : DialectAttribute<(attr
  LocationAttr:$callee,
  LocationAttr:$caller
)>;

//  ///   FileLineColLoc {
//  ///     filename: StringAttr,
//  ///     line: varint,
//  ///     column: varint
//  ///   }
//  kFileLineColLoc = 11,
def FileLineColLoc : DialectAttribute<(attr
  StringAttr:$filename,
  VarInt:$line,
  VarInt:$column
)>;

let cType = "FusedLoc",
    cBuilder = "cast<FusedLoc>(get<FusedLoc>(context, $_args))" in {
//  ///   FusedLoc {
//  ///     locations: Location[]
//  ///   }
//  kFusedLoc = 12,
def FusedLoc : DialectAttribute<(attr
  Array<Location>:$locations
)> {
  let printerPredicate = "!$_val.getMetadata()";
}

//  ///   FusedLocWithMetadata {
//  ///     locations: LocationAttr[],
//  ///     metadata: Attribute
//  ///   }
//  /// A variant of FusedLoc with metadata.
//  kFusedLocWithMetadata = 13,
def FusedLocWithMetadata : DialectAttribute<(attr
  Array<Location>:$locations,
  Attribute:$metadata
)> {
  let printerPredicate = "$_val.getMetadata()";
}
}

//  ///   NameLoc {
//  ///     name: StringAttr,
//  ///     childLoc: LocationAttr
//  ///   }
//  kNameLoc = 14,
def NameLoc : DialectAttribute<(attr
  StringAttr:$name,
  LocationAttr:$childLoc
)>;

//  ///   UnknownLoc {
//  ///   }
//  kUnknownLoc = 15,
def UnknownLoc : DialectAttribute<(attr)>;

//  ///   DenseResourceElementsAttr {
//  ///     type: ShapedType,
//  ///     handle: ResourceHandle
//  ///   }
//  kDenseResourceElementsAttr = 16,
def DenseResourceElementsAttr : DialectAttribute<(attr
  ShapedType:$type,
  ResourceHandle<"DenseResourceElementsHandle">:$rawHandle
)> {
  // Note: order of serialization does not match order of builder.
  let cBuilder = "get<$_resultType>(context, type, *rawHandle)";
}

let cType = "RankedTensorType" in {
//   ///   RankedTensorType {
//   ///     shape: svarint[],
//   ///     elementType: Type,
//   ///   }
//   ///
//   kRankedTensorType = 13,
def RankedTensorType : DialectType<(type
  Array<SignedVarInt>:$shape,
  Type:$elementType
)> {
  let printerPredicate = "!$_val.getEncoding()";
}

//   ///   RankedTensorTypeWithEncoding {
//   ///     encoding: Attribute,
//   ///     shape: svarint[],
//   ///     elementType: Type
//   ///   }
//   /// Variant of RankedTensorType with an encoding.
//   kRankedTensorTypeWithEncoding = 14,
def RankedTensorTypeWithEncoding : DialectType<(type
  Attribute:$encoding,
  Array<SignedVarInt>:$shape,
  Type:$elementType
)> {
  let printerPredicate = "$_val.getEncoding()";
  // Note: order of serialization does not match order of builder.
  let cBuilder = "get<$_resultType>(context, shape, elementType, encoding)";
}
}

//  ///   DenseArrayAttr {
//  ///     elementType: Type,
//  ///     size: varint
//  ///     data: blob
//  ///   }
//  kDenseArrayAttr = 17,
def DenseArrayAttr : DialectAttribute<(attr
  Type:$elementType,
  VarInt:$size,
  Blob:$rawData
)>;

//  ///   DenseIntOrFPElementsAttr {
//  ///     type: ShapedType,
//  ///     data: blob
//  ///   }
//  kDenseIntOrFPElementsAttr = 18,
def DenseElementsAttr : WithType<"DenseElementsAttr", Attribute>;
def DenseIntOrFPElementsAttr : DialectAttribute<(attr
  ShapedType:$type,
  Blob:$rawData
)> {
  let cBuilder = "cast<$_resultType>($_resultType::getFromRawBuffer($_args))";
}

//  ///   DenseStringElementsAttr {
//  ///     type: ShapedType,
//  ///     isSplat: varint,
//  ///     data: string[]
//  ///   }
//  kDenseStringElementsAttr = 19,
def DenseStringElementsAttr : DialectAttribute<(attr
  ShapedType:$type,
  WithGetter<"$_attrType.isSplat()", VarInt>:$_isSplat,
  WithBuilder<"$_args",
    WithType<"SmallVector<StringRef>",
    WithParser <"succeeded(readPotentiallySplatString($_reader, type, _isSplat, $_var))",
    WithPrinter<"writePotentiallySplatString($_writer, $_name)">>>>:$rawStringData
)>;

//  ///   SparseElementsAttr {
//  ///     type: ShapedType,
//  ///     indices: DenseIntElementsAttr,
//  ///     values: DenseElementsAttr
//  ///   }
//  kSparseElementsAttr = 20,
def DenseIntElementsAttr : WithType<"DenseIntElementsAttr", Attribute>;
def SparseElementsAttr : DialectAttribute<(attr
  ShapedType:$type,
  DenseIntElementsAttr:$indices,
  DenseElementsAttr:$values
)>;

// Types
// -----

// enum TypeCode {
//   ///   IntegerType {
//   ///     widthAndSignedness: varint // (width << 2) | (signedness)
//   ///   }
//   ///
//   kIntegerType = 0,
def IntegerType : DialectType<(type
  // Yes not pretty,
  WithParser<"succeeded($_reader.readVarInt($_var))",
    WithBuilder<"$_args",
    WithPrinter<"$_writer.writeVarInt($_name.getWidth() << 2 | $_name.getSignedness())",
    WithType   <"uint64_t">>>>:$_widthAndSignedness,
  // Split up parsed varint for create method.
  LocalVar<"uint64_t", "_widthAndSignedness >> 2">:$width,
  LocalVar<"IntegerType::SignednessSemantics",
    "static_cast<IntegerType::SignednessSemantics>(_widthAndSignedness & 0x3)">:$signedness
)>;

//
//   ///   IndexType {
//   ///   }
//   ///
//   kIndexType = 1,
def IndexType : DialectType<(type)>;

//   ///   FunctionType {
//   ///     inputs: Type[],
//   ///     results: Type[]
//   ///   }
//   ///
//   kFunctionType = 2,
def FunctionType : DialectType<(type
  Array<Type>:$inputs,
  Array<Type>:$results
)>;

//   ///   BFloat16Type {
//   ///   }
//   ///
//   kBFloat16Type = 3,
def BFloat16Type : DialectType<(type)>;

//   ///   Float16Type {
//   ///   }
//   ///
//   kFloat16Type = 4,
def Float16Type : DialectType<(type)>;

//   ///   Float32Type {
//   ///   }
//   ///
//   kFloat32Type = 5,
def Float32Type : DialectType<(type)>;

//   ///   Float64Type {
//   ///   }
//   ///
//   kFloat64Type = 6,
def Float64Type : DialectType<(type)>;

//   ///   Float80Type {
//   ///   }
//   ///
//   kFloat80Type = 7,
def Float80Type : DialectType<(type)>;

//   ///   Float128Type {
//   ///   }
//   ///
//   kFloat128Type = 8,
def Float128Type : DialectType<(type)>;

//   ///   ComplexType {
//   ///     elementType: Type
//   ///   }
//   ///
//   kComplexType = 9,
def ComplexType : DialectType<(type
  Type:$elementType
)>;

def MemRefLayout: WithType<"MemRefLayoutAttrInterface", Attribute>;

let cType = "MemRefType" in {
//   ///   MemRefType {
//   ///     shape: svarint[],
//   ///     elementType: Type,
//   ///     layout: Attribute
//   ///   }
//   ///
//   kMemRefType = 10,
def MemRefType : DialectType<(type
  Array<SignedVarInt>:$shape,
  Type:$elementType,
  MemRefLayout:$layout
)> {
  let printerPredicate = "!$_val.getMemorySpace()";
}

//   ///   MemRefTypeWithMemSpace {
//   ///     memorySpace: Attribute,
//   ///     shape: svarint[],
//   ///     elementType: Type,
//   ///     layout: Attribute
//   ///   }
//   /// Variant of MemRefType with non-default memory space.
//   kMemRefTypeWithMemSpace = 11,
def MemRefTypeWithMemSpace : DialectType<(type
  Attribute:$memorySpace,
  Array<SignedVarInt>:$shape,
  Type:$elementType,
  MemRefLayout:$layout
)> {
  let printerPredicate = "!!$_val.getMemorySpace()";
  // Note: order of serialization does not match order of builder.
  let cBuilder = "get<$_resultType>(context, shape, elementType, layout, memorySpace)";
}
}

//   ///   NoneType {
//   ///   }
//   ///
//   kNoneType = 12,
def NoneType : DialectType<(type)>;

//   ///   TupleType {
//   ///     elementTypes: Type[]
//   ///   }
//   kTupleType = 15,
def TupleType : DialectType<(type
  Array<Type>:$types
)>;

let cType = "UnrankedMemRefType" in {
//   ///   UnrankedMemRefType {
//   ///     elementType: Type
//   ///   }
//   ///
//   kUnrankedMemRefType = 16,
def UnrankedMemRefType : DialectType<(type
  Type:$elementType
)> {
  let printerPredicate = "!$_val.getMemorySpace()";
  let cBuilder = "get<$_resultType>(context, elementType, Attribute())";
}

//   ///   UnrankedMemRefTypeWithMemSpace {
//   ///     memorySpace: Attribute,
//   ///     elementType: Type
//   ///   }
//   /// Variant of UnrankedMemRefType with non-default memory space.
//   kUnrankedMemRefTypeWithMemSpace = 17,
def UnrankedMemRefTypeWithMemSpace : DialectType<(type
  Attribute:$memorySpace,
  Type:$elementType
)> {
  let printerPredicate = "$_val.getMemorySpace()";
  // Note: order of serialization does not match order of builder.
  let cBuilder = "get<$_resultType>(context, elementType, memorySpace)";
}
}

//   ///   UnrankedTensorType {
//   ///     elementType: Type
//   ///   }
//   ///
//   kUnrankedTensorType = 18,
def UnrankedTensorType : DialectType<(type
  Type:$elementType
)>;

let cType = "VectorType" in {
//   ///   VectorType {
//   ///     shape: svarint[],
//   ///     elementType: Type
//   ///   }
//   ///
//   kVectorType = 19,
def VectorType : DialectType<(type
  Array<SignedVarInt>:$shape,
  Type:$elementType
)> {
  let printerPredicate = "!$_val.getNumScalableDims()";
}

//   ///   VectorTypeWithScalableDims {
//   ///     numScalableDims: varint,
//   ///     shape: svarint[],
//   ///     elementType: Type
//   ///   }
//   /// Variant of VectorType with scalable dimensions.
//   kVectorTypeWithScalableDims = 20,
def VectorTypeWithScalableDims : DialectType<(type
  VarInt:$numScalableDims,
  Array<SignedVarInt>:$shape,
  Type:$elementType
)> {
  let printerPredicate = "$_val.getNumScalableDims()";
  // Note: order of serialization does not match order of builder.
  let cBuilder = "get<$_resultType>(context, shape, elementType, numScalableDims)";
}
}

/// This enum contains marker codes used to indicate which attribute is
/// currently being decoded, and how it should be decoded. The order of these
/// codes should generally be unchanged, as any changes will inevitably break
/// compatibility with older bytecode.

def BuiltinDialectAttributes : DialectAttributes<"Builtin"> {
  let elems = [
    ArrayAttr,
    DictionaryAttr,
    StringAttr,
    StringAttrWithType,
    FlatSymbolRefAttr,
    SymbolRefAttr,
    TypeAttr,
    UnitAttr,
    IntegerAttr,
    FloatAttr,
    CallSiteLoc,
    FileLineColLoc,
    FusedLoc,
    FusedLocWithMetadata,
    NameLoc,
    UnknownLoc,
    DenseResourceElementsAttr,
    DenseArrayAttr,
    DenseIntOrFPElementsAttr,
    DenseStringElementsAttr,
    SparseElementsAttr
  ];
}

def BuiltinDialectTypes : DialectTypes<"Builtin"> {
  let elems = [
    IntegerType,
    IndexType,
    FunctionType,
    BFloat16Type,
    Float16Type,
    Float32Type,
    Float64Type,
    Float80Type,
    Float128Type,
    ComplexType,
    MemRefType,
    MemRefTypeWithMemSpace,
    NoneType,
    RankedTensorType,
    RankedTensorTypeWithEncoding,
    TupleType,
    UnrankedMemRefType,
    UnrankedMemRefTypeWithMemSpace,
    UnrankedTensorType,
    VectorType,
    VectorTypeWithScalableDims
  ];
}

#endif // BUILTIN_BYTECODE
