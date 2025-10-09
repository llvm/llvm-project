//===-- LLVMDialectBytecode.td - LLVM bytecode defs --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the LLVM bytecode reader/writer definition file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DIALECT_BYTECODE
#define LLVM_DIALECT_BYTECODE

include "mlir/IR/BytecodeBase.td"

//===----------------------------------------------------------------------===//
// Bytecode classes for attributes and types.
//===----------------------------------------------------------------------===//

def String :
  WithParser <"succeeded($_reader.readString($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeOwnedString($_getter)",
  WithType   <"StringRef">>>>;

class Attr<string type> : WithType<type, Attribute>;

class OptionalAttribute<string type> :
  WithParser <"succeeded($_reader.readOptionalAttribute($_var))",
  WithPrinter<"$_writer.writeOptionalAttribute($_getter)",
  WithType<type, Attribute>>>;

class OptionalInt<string type> :
  WithParser <"succeeded(readOptionalInt($_reader, $_var))",
  WithPrinter<"writeOptionalInt($_writer, $_getter)",
  WithType<"std::optional<" # type # ">", VarInt>>>;

class OptionalArrayRef<string eltType> :
  WithParser <"succeeded(readOptionalArrayRef<"
    # eltType # ">($_reader, $_var))",
  WithPrinter<"writeOptionalArrayRef<"
    # eltType # ">($_writer, $_getter)",
  WithType<"SmallVector<"
    # eltType # ">", Attribute>>>;

class EnumClassFlag<string flag, string getter> :
    WithParser<"succeeded($_reader.readVarInt($_var))",
    WithBuilder<"(" # flag # ")$_args",
    WithPrinter<"$_writer.writeVarInt((uint64_t)$_name." # getter # ")",
    WithType<"uint64_t", VarInt>>>>;

//===----------------------------------------------------------------------===//
// General notes
// - For each attribute or type entry, the argument names should match
//   LLVMAttrDefs.td
// - The mnemonics are either LLVM or builtin MLIR attributes and types, but
//   regular C++ types are also allowed to match builders and parsers.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AliasScopeAttr
//===----------------------------------------------------------------------===//

def AliasScopeAttr : DialectAttribute<(attr
  Attr<"Attribute">:$id,
  Attr<"AliasScopeDomainAttr">:$domain,
  OptionalAttribute<"StringAttr">:$description
)>;

//===----------------------------------------------------------------------===//
// DIBasicTypeAttr
//===----------------------------------------------------------------------===//

def DIBasicTypeAttr : DialectAttribute<(attr
  VarInt:$tag,
  String:$name,
  VarInt:$sizeInBits,
  VarInt:$encoding
)>;

//===----------------------------------------------------------------------===//
// DIExpressionAttr, DIExpressionElemAttr
//===----------------------------------------------------------------------===//

def DIExpressionElemAttr : DialectAttribute<(attr
  VarInt:$opcode,
  OptionalArrayRef<"uint64_t">:$arguments
)>;

def DIExpressionAttr : DialectAttribute<(attr
  OptionalArrayRef<"DIExpressionElemAttr">:$operations
)>;

//===----------------------------------------------------------------------===//
// DIFileAttr
//===----------------------------------------------------------------------===//

def DIFileAttr : DialectAttribute<(attr
  String:$name,
  String:$directory
)>;

//===----------------------------------------------------------------------===//
// DILocalVariableAttr
//===----------------------------------------------------------------------===//

def DILocalVariableAttr : DialectAttribute<(attr
  Attr<"DIScopeAttr">:$scope, // Non-optional attribute
  OptionalAttribute<"StringAttr">:$name,
  OptionalAttribute<"DIFileAttr">:$file,
  VarInt:$line,
  VarInt:$arg,
  VarInt:$alignInBits,
  OptionalAttribute<"DITypeAttr">:$type,
  EnumClassFlag<"DIFlags", "getFlags()">:$_rawflags,
  LocalVar<"DIFlags", "(DIFlags)_rawflags">:$flags
)> {
  // DILocalVariableAttr direct getter uses a `StringRef` for `name`. Since the
  // more direct getter is prefered during bytecode reading, force the base one
  // and prevent crashes for empty `StringAttr`.
  let cBuilder = "$_resultType::get(context, $_args)";
}

//===----------------------------------------------------------------------===//
// DISubroutineTypeAttr
//===----------------------------------------------------------------------===//

def DISubroutineTypeAttr : DialectAttribute<(attr
  VarInt:$callingConvention,
  OptionalArrayRef<"DITypeAttr">:$types
)>;

//===----------------------------------------------------------------------===//
// DICompileUnitAttr
//===----------------------------------------------------------------------===//

def DICompileUnitAttr : DialectAttribute<(attr
  Attr<"DistinctAttr">:$id,
  VarInt:$sourceLanguage,
  Attr<"DIFileAttr">:$file,
  OptionalAttribute<"StringAttr">:$producer,
  Bool:$isOptimized,
  EnumClassFlag<"DIEmissionKind", "getEmissionKind()">:$_rawEmissionKind,
  LocalVar<"DIEmissionKind", "(DIEmissionKind)_rawEmissionKind">:$emissionKind,
  EnumClassFlag<"DINameTableKind", "getNameTableKind()">:$_rawNameTableKind,
  LocalVar<"DINameTableKind",
           "(DINameTableKind)_rawNameTableKind">:$nameTableKind
)>;

//===----------------------------------------------------------------------===//
// DISubprogramAttr
//===----------------------------------------------------------------------===//

def DISubprogramAttr : DialectAttribute<(attr
  OptionalAttribute<"DistinctAttr">:$recId,
  Bool:$isRecSelf,
  OptionalAttribute<"DistinctAttr">:$id,
  OptionalAttribute<"DICompileUnitAttr">:$compileUnit,
  OptionalAttribute<"DIScopeAttr">:$scope, // TODO: DIScopeAttr
  OptionalAttribute<"StringAttr">:$name,
  OptionalAttribute<"StringAttr">:$linkageName,
  OptionalAttribute<"DIFileAttr">:$file,
  VarInt:$line,
  VarInt:$scopeLine,
  EnumClassFlag<"DISubprogramFlags", "getSubprogramFlags()">:$_rawflags,
  LocalVar<"DISubprogramFlags", "(DISubprogramFlags)_rawflags">:$subprogramFlags,
  OptionalAttribute<"DISubroutineTypeAttr">:$type,
  OptionalArrayRef<"DINodeAttr">:$retainedNodes,
  OptionalArrayRef<"DINodeAttr">:$annotations
)>;

//===----------------------------------------------------------------------===//
// DICompositeTypeAttr
//===----------------------------------------------------------------------===//

def DICompositeTypeAttr : DialectAttribute<(attr
  OptionalAttribute<"DistinctAttr">:$recId,
  Bool:$isRecSelf,
  VarInt:$tag,
  OptionalAttribute<"StringAttr">:$name,
  OptionalAttribute<"DIFileAttr">:$file,
  VarInt:$line,
  OptionalAttribute<"DIScopeAttr">:$scope,
  OptionalAttribute<"DITypeAttr">:$baseType,
  EnumClassFlag<"DIFlags", "getFlags()">:$_rawflags,
  LocalVar<"DIFlags", "(DIFlags)_rawflags">:$flags,
  VarInt:$sizeInBits,
  VarInt:$alignInBits,
  OptionalAttribute<"DIExpressionAttr">:$dataLocation,
  OptionalAttribute<"DIExpressionAttr">:$rank,
  OptionalAttribute<"DIExpressionAttr">:$allocated,
  OptionalAttribute<"DIExpressionAttr">:$associated,
  OptionalArrayRef<"DINodeAttr">:$elements
)>;

//===----------------------------------------------------------------------===//
// DIDerivedTypeAttr
//===----------------------------------------------------------------------===//

def DIDerivedTypeAttr : DialectAttribute<(attr
  VarInt:$tag,
  OptionalAttribute<"StringAttr">:$name,
  OptionalAttribute<"DITypeAttr">:$baseType,
  VarInt:$sizeInBits,
  VarInt:$alignInBits,
  VarInt:$offsetInBits,
  OptionalInt<"unsigned">:$dwarfAddressSpace,
  OptionalAttribute<"DINodeAttr">:$extraData // TODO: DINodeAttr
)>;

//===----------------------------------------------------------------------===//
// DIImportedEntityAttr
//===----------------------------------------------------------------------===//

def DIImportedEntityAttr : DialectAttribute<(attr
  VarInt:$tag,
  Attr<"DIScopeAttr">:$scope,
  Attr<"DINodeAttr">:$entity,
  OptionalAttribute<"DIFileAttr">:$file,
  VarInt:$line,
  OptionalAttribute<"StringAttr">:$name,
  OptionalArrayRef<"DINodeAttr">:$elements
)>;

//===----------------------------------------------------------------------===//
// DIGlobalVariableAttr, DIGlobalVariableExpressionAttr
//===----------------------------------------------------------------------===//

def DIGlobalVariableAttr : DialectAttribute<(attr
  OptionalAttribute<"DIScopeAttr">:$scope,
  OptionalAttribute<"StringAttr">:$name,
  OptionalAttribute<"StringAttr">:$linkageName,
  Attr<"DIFileAttr">:$file,
  VarInt:$line,
  Attr<"DITypeAttr">:$type,
  Bool:$isLocalToUnit,
  Bool:$isDefined,
  VarInt:$alignInBits
)>;

def DIGlobalVariableExpressionAttr : DialectAttribute<(attr
  Attr<"DIGlobalVariableAttr">:$var,
  OptionalAttribute<"DIExpressionAttr">:$expr
)>;

//===----------------------------------------------------------------------===//
// DILabelAttr
//===----------------------------------------------------------------------===//

def DILabelAttr : DialectAttribute<(attr
  Attr<"DIScopeAttr">:$scope,
  OptionalAttribute<"StringAttr">:$name,
  OptionalAttribute<"DIFileAttr">:$file,
  VarInt:$line
)> {
  // DILabelAttr direct getter uses a `StringRef` for `name`. Since the
  // more direct getter is prefered during bytecode reading, force the base one
  // and prevent crashes for empty `StringAttr`.
  let cBuilder = "$_resultType::get(context, $_args)";
}

//===----------------------------------------------------------------------===//
// DILexicalBlockAttr, DILexicalBlockFileAttr
//===----------------------------------------------------------------------===//

def DILexicalBlockAttr : DialectAttribute<(attr
  Attr<"DIScopeAttr">:$scope,
  OptionalAttribute<"DIFileAttr">:$file,
  VarInt:$line,
  VarInt:$column
)>;

def DILexicalBlockFileAttr : DialectAttribute<(attr
  Attr<"DIScopeAttr">:$scope,
  OptionalAttribute<"DIFileAttr">:$file,
  VarInt:$discriminator
)>;

//===----------------------------------------------------------------------===//
// DINamespaceAttr
//===----------------------------------------------------------------------===//

def DINamespaceAttr : DialectAttribute<(attr
  OptionalAttribute<"StringAttr">:$name,
  OptionalAttribute<"DIScopeAttr">:$scope,
  Bool:$exportSymbols
)>;

//===----------------------------------------------------------------------===//
// DISubrangeAttr
//===----------------------------------------------------------------------===//

def DISubrangeAttr : DialectAttribute<(attr
  OptionalAttribute<"Attribute">:$count,
  OptionalAttribute<"Attribute">:$lowerBound,
  OptionalAttribute<"Attribute">:$upperBound,
  OptionalAttribute<"Attribute">:$stride
)>;

//===----------------------------------------------------------------------===//
// LoopAnnotationAttr
//===----------------------------------------------------------------------===//

def LoopAnnotationAttr : DialectAttribute<(attr
  OptionalAttribute<"BoolAttr">:$disableNonforced,
  OptionalAttribute<"LoopVectorizeAttr">:$vectorize, // TODO: LoopVectorizeAttr
  OptionalAttribute<"LoopInterleaveAttr">:$interleave, // TODO: LoopInterleaveAttr
  OptionalAttribute<"LoopUnrollAttr">:$unroll, // TODO: LoopUnrollAttr
  OptionalAttribute<"LoopUnrollAndJamAttr">:$unrollAndJam, // TODO: LoopUnrollAndJamAttr
  OptionalAttribute<"LoopLICMAttr">:$licm, // TODO: LoopLICMAttr
  OptionalAttribute<"LoopDistributeAttr">:$distribute, // TODO: LoopDistributeAttr
  OptionalAttribute<"LoopPipelineAttr">:$pipeline, // TODO: LoopPipelineAttr
  OptionalAttribute<"LoopPeeledAttr">:$peeled, // TODO: LoopPeeledAttr
  OptionalAttribute<"LoopUnswitchAttr">:$unswitch, // TODO: LoopUnswitchAttr
  OptionalAttribute<"BoolAttr">:$mustProgress,
  OptionalAttribute<"BoolAttr">:$isVectorized,
  OptionalAttribute<"FusedLoc">:$startLoc,
  OptionalAttribute<"FusedLoc">:$endLoc,
  OptionalArrayRef<"AccessGroupAttr">:$parallelAccesses // TODO: AccessGroupAttr
)>;

//===----------------------------------------------------------------------===//
// Attributes & Types with custom bytecode handling.
//===----------------------------------------------------------------------===//

def LLVMDialectAttributes : DialectAttributes<"LLVM"> {
  let elems = [
    AliasScopeAttr,
    DIBasicTypeAttr,
    DICompileUnitAttr,
    DICompositeTypeAttr,
    DIDerivedTypeAttr,
    DIExpressionElemAttr,
    DIExpressionAttr,
    DIFileAttr,
    DIGlobalVariableAttr,
    DIGlobalVariableExpressionAttr,
    DIImportedEntityAttr,
    DILabelAttr,
    DILexicalBlockAttr,
    DILexicalBlockFileAttr,
    DILocalVariableAttr,
    DINamespaceAttr,
    DISubprogramAttr,
    DISubrangeAttr,
    DISubroutineTypeAttr,
    LoopAnnotationAttr
  ];
}

def LLVMDialectTypes : DialectTypes<"LLVM"> {
  let elems = [];
}

#endif // LLVM_DIALECT_BYTECODE
