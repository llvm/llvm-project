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

def ArrayAttr : DialectAttribute<(attr
  Array<Attribute>:$value
)>;

let cType = "StringAttr" in {
def StringAttr : DialectAttribute<(attr
  String:$value
)> {
  let printerPredicate = "::llvm::isa<NoneType>($_val.getType())";
}

def StringAttrWithType : DialectAttribute<(attr
  String:$value,
  Type:$type
)> { let printerPredicate = "!::llvm::isa<NoneType>($_val.getType())"; }
}

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

def FlatSymbolRefAttr: DialectAttribute<(attr
  StringAttr:$rootReference
)>;

def SymbolRefAttr: DialectAttribute<(attr
  StringAttr:$rootReference,
  Array<FlatSymbolRefAttr>:$nestedReferences
)>;

def TypeAttr: DialectAttribute<(attr
  Type:$value
)>;

def UnitAttr: DialectAttribute<(attr)>;

def IntegerAttr: DialectAttribute<(attr
  Type:$type,
  KnownWidthAPInt<"type">:$value
)> {
  let cBuilder = "get<$_resultType>(context, type, *value)";
}

defvar FloatType = Type;
def FloatAttr : DialectAttribute<(attr
  FloatType:$type,
  KnownSemanticsAPFloat<"type">:$value
)> {
  let cBuilder = "get<$_resultType>(context, type, *value)";
}

def CallSiteLoc : DialectAttribute<(attr
  LocationAttr:$callee,
  LocationAttr:$caller
)>;

def FileLineColLoc : DialectAttribute<(attr
  StringAttr:$filename,
  VarInt:$line,
  VarInt:$column
)>;

let cType = "FusedLoc",
    cBuilder = "cast<FusedLoc>(get<FusedLoc>(context, $_args))" in {
def FusedLoc : DialectAttribute<(attr
  Array<Location>:$locations
)> {
  let printerPredicate = "!$_val.getMetadata()";
}

def FusedLocWithMetadata : DialectAttribute<(attr
  Array<Location>:$locations,
  Attribute:$metadata
)> {
  let printerPredicate = "$_val.getMetadata()";
}
}

def NameLoc : DialectAttribute<(attr
  StringAttr:$name,
  LocationAttr:$childLoc
)>;

def UnknownLoc : DialectAttribute<(attr)>;

def DenseResourceElementsAttr : DialectAttribute<(attr
  ShapedType:$type,
  ResourceHandle<"DenseResourceElementsHandle">:$rawHandle
)> {
  // Note: order of serialization does not match order of builder.
  let cBuilder = "get<$_resultType>(context, type, *rawHandle)";
}

let cType = "RankedTensorType" in {
def RankedTensorType : DialectType<(type
  Array<SignedVarIntList>:$shape,
  Type:$elementType
)> {
  let printerPredicate = "!$_val.getEncoding()";
}

def RankedTensorTypeWithEncoding : DialectType<(type
  Attribute:$encoding,
  Array<SignedVarIntList>:$shape,
  Type:$elementType
)> {
  let printerPredicate = "$_val.getEncoding()";
  // Note: order of serialization does not match order of builder.
  let cBuilder = "get<$_resultType>(context, shape, elementType, encoding)";
}
}

def DenseArrayAttr : DialectAttribute<(attr
  Type:$elementType,
  VarInt:$size,
  Blob:$rawData
)>;

def DenseElementsAttr : WithType<"DenseElementsAttr", Attribute>;
def DenseIntOrFPElementsAttr : DialectAttribute<(attr
  ShapedType:$type,
  Blob:$rawData
)> {
  let cBuilder = "cast<$_resultType>($_resultType::getFromRawBuffer($_args))";
}

def DenseStringElementsAttr : DialectAttribute<(attr
  ShapedType:$type,
  WithGetter<"$_attrType.isSplat()", VarInt>:$_isSplat,
  WithBuilder<"$_args",
    WithType<"SmallVector<StringRef>",
    WithParser <"succeeded(readPotentiallySplatString($_reader, type, _isSplat, $_var))",
    WithPrinter<"writePotentiallySplatString($_writer, $_name)">>>>:$rawStringData
)>;

def DenseIntElementsAttr : WithType<"DenseIntElementsAttr", Attribute>;
def SparseElementsAttr : DialectAttribute<(attr
  ShapedType:$type,
  DenseIntElementsAttr:$indices,
  DenseElementsAttr:$values
)>;

def DistinctAttr : DialectAttribute<(attr
  Attribute:$referencedAttr
)>;

// Types
// -----

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

def IndexType : DialectType<(type)>;

def FunctionType : DialectType<(type
  Array<Type>:$inputs,
  Array<Type>:$results
)>;

def BFloat16Type : DialectType<(type)>;

def Float16Type : DialectType<(type)>;

def Float32Type : DialectType<(type)>;

def Float64Type : DialectType<(type)>;

def Float80Type : DialectType<(type)>;

def Float128Type : DialectType<(type)>;

def ComplexType : DialectType<(type
  Type:$elementType
)>;

def MemRefLayout: WithType<"MemRefLayoutAttrInterface", Attribute>;

let cType = "MemRefType" in {
def MemRefType : DialectType<(type
  Array<SignedVarIntList>:$shape,
  Type:$elementType,
  MemRefLayout:$layout
)> {
  let printerPredicate = "!$_val.getMemorySpace()";
}

def MemRefTypeWithMemSpace : DialectType<(type
  Attribute:$memorySpace,
  Array<SignedVarIntList>:$shape,
  Type:$elementType,
  MemRefLayout:$layout
)> {
  let printerPredicate = "!!$_val.getMemorySpace()";
  // Note: order of serialization does not match order of builder.
  let cBuilder = "get<$_resultType>(context, shape, elementType, layout, memorySpace)";
}
}

def NoneType : DialectType<(type)>;

def TupleType : DialectType<(type
  Array<Type>:$types
)>;

let cType = "UnrankedMemRefType" in {
def UnrankedMemRefType : DialectType<(type
  Type:$elementType
)> {
  let printerPredicate = "!$_val.getMemorySpace()";
  let cBuilder = "get<$_resultType>(context, elementType, Attribute())";
}

def UnrankedMemRefTypeWithMemSpace : DialectType<(type
  Attribute:$memorySpace,
  Type:$elementType
)> {
  let printerPredicate = "$_val.getMemorySpace()";
  // Note: order of serialization does not match order of builder.
  let cBuilder = "get<$_resultType>(context, elementType, memorySpace)";
}
}

def UnrankedTensorType : DialectType<(type
  Type:$elementType
)>;

let cType = "VectorType" in {
def VectorType : DialectType<(type
  Array<SignedVarIntList>:$shape,
  Type:$elementType
)> {
  let printerPredicate = "!$_val.isScalable()";
}

def VectorTypeWithScalableDims : DialectType<(type
  Array<BoolList>:$scalableDims,
  Array<SignedVarIntList>:$shape,
  Type:$elementType
)> {
  let printerPredicate = "$_val.isScalable()";
  // Note: order of serialization does not match order of builder.
  let cBuilder = "get<$_resultType>(context, shape, elementType, scalableDims)";
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
    SparseElementsAttr,
    DistinctAttr
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
