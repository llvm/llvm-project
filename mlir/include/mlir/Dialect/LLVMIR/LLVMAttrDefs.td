//===-- LLVMAttrDefs.td - LLVM Attributes definition file --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_ATTRDEFS
#define LLVMIR_ATTRDEFS

include "mlir/Dialect/LLVMIR/LLVMDialect.td"
include "mlir/Dialect/LLVMIR/LLVMInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"

// All of the attributes will extend this class.
class LLVM_Attr<string name, string attrMnemonic,
                list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<LLVM_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// CConvAttr
//===----------------------------------------------------------------------===//

def CConvAttr : LLVM_Attr<"CConv", "cconv"> {
  let parameters = (ins "CConv":$CallingConv);
  let assemblyFormat = "`<` $CallingConv `>`";
}

//===----------------------------------------------------------------------===//
// ComdatAttr
//===----------------------------------------------------------------------===//

def ComdatAttr : LLVM_Attr<"Comdat", "comdat"> {
  let parameters = (ins "comdat::Comdat":$comdat);
  let assemblyFormat = "$comdat";
}

//===----------------------------------------------------------------------===//
// LinkageAttr
//===----------------------------------------------------------------------===//

def LinkageAttr : LLVM_Attr<"Linkage", "linkage"> {
  let parameters = (ins "linkage::Linkage":$linkage);
  let assemblyFormat = "`<` $linkage `>`";
}

//===----------------------------------------------------------------------===//
// FramePointerKindAttr
//===----------------------------------------------------------------------===//

def FramePointerKindAttr : LLVM_Attr<"FramePointerKind", "framePointerKind"> {
  let parameters = (ins "framePointerKind::FramePointerKind":$framePointerKind);
  let assemblyFormat = "`<` $framePointerKind `>`";
}

//===----------------------------------------------------------------------===//
// Loop Attributes
//===----------------------------------------------------------------------===//

def LoopVectorizeAttr : LLVM_Attr<"LoopVectorize", "loop_vectorize"> {
  let description = [{
    This attribute defines vectorization specific loop annotations that map to
    the "!llvm.loop.vectorize" metadata.
  }];

  let parameters = (ins
    OptionalParameter<"BoolAttr">:$disable,
    OptionalParameter<"BoolAttr">:$predicateEnable,
    OptionalParameter<"BoolAttr">:$scalableEnable,
    OptionalParameter<"IntegerAttr">:$width,
    OptionalParameter<"LoopAnnotationAttr">:$followupVectorized,
    OptionalParameter<"LoopAnnotationAttr">:$followupEpilogue,
    OptionalParameter<"LoopAnnotationAttr">:$followupAll
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def LoopInterleaveAttr : LLVM_Attr<"LoopInterleave", "loop_interleave"> {
  let description = [{
    This attribute defines interleaving specific loop annotations that map to
    the "!llvm.loop.interleave" metadata.
  }];

  let parameters = (ins
    "IntegerAttr":$count
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def LoopUnrollAttr : LLVM_Attr<"LoopUnroll", "loop_unroll"> {
  let description = [{
    This attribute defines unrolling specific loop annotations that map to
    the "!llvm.loop.unroll" metadata.
  }];

  let parameters = (ins
    OptionalParameter<"BoolAttr">:$disable,
    OptionalParameter<"IntegerAttr">:$count,
    OptionalParameter<"BoolAttr">:$runtimeDisable,
    OptionalParameter<"BoolAttr">:$full,
    OptionalParameter<"LoopAnnotationAttr">:$followupUnrolled,
    OptionalParameter<"LoopAnnotationAttr">:$followupRemainder,
    OptionalParameter<"LoopAnnotationAttr">:$followupAll
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def LoopUnrollAndJamAttr : LLVM_Attr<"LoopUnrollAndJam", "loop_unroll_and_jam"> {
  let description = [{
    This attribute defines "unroll and jam" specific loop annotations that map to
    the "!llvm.loop.unroll_and_jam" metadata.
  }];

  let parameters = (ins
    OptionalParameter<"BoolAttr">:$disable,
    OptionalParameter<"IntegerAttr">:$count,
    OptionalParameter<"LoopAnnotationAttr">:$followupOuter,
    OptionalParameter<"LoopAnnotationAttr">:$followupInner,
    OptionalParameter<"LoopAnnotationAttr">:$followupRemainderOuter,
    OptionalParameter<"LoopAnnotationAttr">:$followupRemainderInner,
    OptionalParameter<"LoopAnnotationAttr">:$followupAll
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def LoopLICMAttr : LLVM_Attr<"LoopLICM", "loop_licm"> {
  let description = [{
    This attribute encapsulates loop invariant code motion (licm) specific loop
    annotations. The fields correspond to the "!llvm.licm.disable" and the
    "!llvm.loop.licm_versioning.disable" metadata.
  }];

  let parameters = (ins
    OptionalParameter<"BoolAttr">:$disable,
    OptionalParameter<"BoolAttr">:$versioningDisable
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def LoopDistributeAttr : LLVM_Attr<"LoopDistribute", "loop_distribute"> {
  let description = [{
    This attribute defines distribution specific loop annotations that map to
    the "!llvm.loop.distribute" metadata.
  }];

  let parameters = (ins
    OptionalParameter<"BoolAttr">:$disable,
    OptionalParameter<"LoopAnnotationAttr">:$followupCoincident,
    OptionalParameter<"LoopAnnotationAttr">:$followupSequential,
    OptionalParameter<"LoopAnnotationAttr">:$followupFallback,
    OptionalParameter<"LoopAnnotationAttr">:$followupAll
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def LoopPipelineAttr : LLVM_Attr<"LoopPipeline", "loop_pipeline"> {
  let description = [{
    This attribute defines pipelining specific loop annotations that map to
    the "!llvm.loop.pipeline" metadata.
  }];

  let parameters = (ins
    OptionalParameter<"BoolAttr">:$disable,
    OptionalParameter<"IntegerAttr">:$initiationinterval
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def LoopPeeledAttr : LLVM_Attr<"LoopPeeled", "loop_peeled"> {
  let description = [{
    This attribute defines pipelining specific loop annotations that map to
    the "!llvm.loop.peeled" metadata.
  }];

  let parameters = (ins
    OptionalParameter<"IntegerAttr">:$count
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def LoopUnswitchAttr : LLVM_Attr<"LoopUnswitch", "loop_unswitch"> {
  let description = [{
    This attribute defines pipelining specific loop annotations that map to
    the "!llvm.loop.unswitch" metadata.
  }];

  let parameters = (ins
    OptionalParameter<"BoolAttr">:$partialDisable
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def LoopAnnotationAttr : LLVM_Attr<"LoopAnnotation", "loop_annotation"> {
  let description = [{
    This attributes encapsulates "loop metadata". It is meant to decorate
    branches that are "latches" (loop backedges) and maps to the `!llvm.loop`
    metadatas: https://llvm.org/docs/LangRef.html#llvm-loop
    It stores annotations in attribute parameters and groups related options in
    nested attributes to provide structured access.
  }];

  let parameters = (ins
    OptionalParameter<"BoolAttr">:$disableNonforced,
    OptionalParameter<"LoopVectorizeAttr">:$vectorize,
    OptionalParameter<"LoopInterleaveAttr">:$interleave,
    OptionalParameter<"LoopUnrollAttr">:$unroll,
    OptionalParameter<"LoopUnrollAndJamAttr">:$unrollAndJam,
    OptionalParameter<"LoopLICMAttr">:$licm,
    OptionalParameter<"LoopDistributeAttr">:$distribute,
    OptionalParameter<"LoopPipelineAttr">:$pipeline,
    OptionalParameter<"LoopPeeledAttr">:$peeled,
    OptionalParameter<"LoopUnswitchAttr">:$unswitch,
    OptionalParameter<"BoolAttr">:$mustProgress,
    OptionalParameter<"BoolAttr">:$isVectorized,
    OptionalParameter<"FusedLoc">:$startLoc,
    OptionalParameter<"FusedLoc">:$endLoc,
    OptionalArrayRefParameter<"AccessGroupAttr">:$parallelAccesses
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DebugInfo Attributes
//===----------------------------------------------------------------------===//

class LLVM_DIParameter<string summary, string default, string parseName,
                       string errorCase, string printName = parseName>
    : AttrOrTypeParameter<"unsigned", "debug info " # summary> {
  let parser = [{ [&]() -> FailureOr<unsigned> {
    SMLoc tagLoc = $_parser.getCurrentLocation();
    StringRef name;
    if ($_parser.parseKeyword(&name))
      return failure();

    unsigned tag = llvm::dwarf::get}] # parseName # [{(name);
    if (tag == }] # errorCase # [{)
      return $_parser.emitError(tagLoc)
        << "invalid debug info }] # summary # [{ name: " << name;
    return tag;
  }() }];
  let printer = "$_printer << llvm::dwarf::" # printName # "String($_self)";
  let defaultValue = default;
}

def LLVM_DICallingConventionParameter : LLVM_DIParameter<
  "calling convention", /*default=*/"0", "CallingConvention", /*errorCase=*/"0",
  "Convention"
>;

def LLVM_DIEncodingParameter : LLVM_DIParameter<
  "encoding", /*default=*/"0", "AttributeEncoding", /*errorCase=*/"0"
>;

def LLVM_DILanguageParameter : LLVM_DIParameter<
  "language", /*default=*/"", "Language", /*errorCase=*/"0"
>;

def LLVM_DITagParameter : LLVM_DIParameter<
  "tag", /*default=*/"0", "Tag", /*errorCase=*/"llvm::dwarf::DW_TAG_invalid"
>;

def LLVM_DIOperationEncodingParameter : LLVM_DIParameter<
  "operation encoding", /*default=*/"", "OperationEncoding", /*errorCase=*/"0"
>;

//===----------------------------------------------------------------------===//
// DIExpressionAttr
//===----------------------------------------------------------------------===//

def LLVM_DIExpressionElemAttr : LLVM_Attr<"DIExpressionElem",
                                          "di_expression_elem"> {
  let parameters = (ins
    LLVM_DIOperationEncodingParameter:$opcode,
    OptionalArrayRefParameter<"uint64_t">:$arguments);
  let assemblyFormat = [{
    `` $opcode ( `(` custom<ExpressionArg>(ref($opcode), $arguments)^ `)` ) : (``)?
  }];
}

def LLVM_DIExpressionAttr : LLVM_Attr<"DIExpression", "di_expression"> {
  let parameters = (ins
    OptionalArrayRefParameter<"DIExpressionElemAttr">:$operations
  );
  let builders = [
    AttrBuilder<(ins)>
  ];
  let constBuilderCall =
            "::mlir::LLVM::DIExpressionAttr::get($_builder.getContext(), $0)";
  let assemblyFormat = "`<` ( `[` $operations^ `]` ) : (``)? `>`";
}

//===----------------------------------------------------------------------===//
// DINullTypeAttr
//===----------------------------------------------------------------------===//

def LLVM_DINullTypeAttr : LLVM_Attr<"DINullType", "di_null_type",
                                    /*traits=*/[], "DITypeAttr"> {
  let parameters = (ins);
}

//===----------------------------------------------------------------------===//
// DIBasicTypeAttr
//===----------------------------------------------------------------------===//

def LLVM_DIBasicTypeAttr : LLVM_Attr<"DIBasicType", "di_basic_type",
                                     /*traits=*/[], "DITypeAttr"> {
  let parameters = (ins
    LLVM_DITagParameter:$tag,
    "StringAttr":$name,
    OptionalParameter<"uint64_t">:$sizeInBits,
    LLVM_DIEncodingParameter:$encoding
  );

  let builders = [
    TypeBuilder<(ins
      "unsigned":$tag, "const Twine &":$name, "uint64_t":$sizeInBits,
      "unsigned":$encoding
    ), [{
      return $_get($_ctxt, tag, StringAttr::get($_ctxt, name), sizeInBits,
                   encoding);
    }]>
  ];
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DICompileUnitAttr
//===----------------------------------------------------------------------===//

def LLVM_DICompileUnitAttr : LLVM_Attr<"DICompileUnit", "di_compile_unit",
                                       /*traits=*/[], "DIScopeAttr"> {
  let parameters = (ins
    "DistinctAttr":$id,
    LLVM_DILanguageParameter:$sourceLanguage,
    "DIFileAttr":$file,
    OptionalParameter<"StringAttr">:$producer,
    "bool":$isOptimized,
    "DIEmissionKind":$emissionKind,
    OptionalParameter<"DINameTableKind">:$nameTableKind
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins
      "DistinctAttr":$id, "unsigned":$sourceLanguage, "DIFileAttr":$file,
      "StringAttr":$producer, "bool":$isOptimized,
      "DIEmissionKind":$emissionKind,
      CArg<"DINameTableKind", "DINameTableKind::Default">:$nameTableKind
    ), [{
      return $_get(id.getContext(), id, sourceLanguage, file, producer,
                   isOptimized, emissionKind, nameTableKind);
    }]>
  ];
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DICompositeTypeAttr
//===----------------------------------------------------------------------===//

def LLVM_DICompositeTypeAttr : LLVM_Attr<"DICompositeType", "di_composite_type",
                                         [LLVM_DIRecursiveTypeAttrInterface],
                                         "DITypeAttr"> {
  let parameters = (ins
    // DIRecursiveTypeAttrInterface specific parameters.
    OptionalParameter<"DistinctAttr">:$recId,
    OptionalParameter<"bool">:$isRecSelf,
    // DICompositeType specific parameters.
    LLVM_DITagParameter:$tag,
    OptionalParameter<"StringAttr">:$name,
    OptionalParameter<"DIFileAttr">:$file,
    OptionalParameter<"uint32_t">:$line,
    OptionalParameter<"DIScopeAttr">:$scope,
    OptionalParameter<"DITypeAttr">:$baseType,
    OptionalParameter<"DIFlags">:$flags,
    OptionalParameter<"uint64_t">:$sizeInBits,
    OptionalParameter<"uint64_t">:$alignInBits,
    OptionalArrayRefParameter<"DINodeAttr">:$elements,
    OptionalParameter<"DIExpressionAttr">:$dataLocation,
    OptionalParameter<"DIExpressionAttr">:$rank,
    OptionalParameter<"DIExpressionAttr">:$allocated,
    OptionalParameter<"DIExpressionAttr">:$associated
  );
  let builders = [
    AttrBuilder<(ins
      "unsigned":$tag, "StringAttr":$name, "DIFileAttr":$file,
      "uint32_t":$line, "DIScopeAttr":$scope, "DITypeAttr":$baseType,
      "DIFlags":$flags, "uint64_t":$sizeInBits, "uint64_t":$alignInBits,
      "ArrayRef<DINodeAttr>":$elements, "DIExpressionAttr":$dataLocation,
      "DIExpressionAttr":$rank, "DIExpressionAttr":$allocated,
      "DIExpressionAttr":$associated
    ), [{
      return $_get($_ctxt, /*recId=*/nullptr, /*isRecSelf=*/nullptr,
                   tag, name, file, line, scope, baseType, flags, sizeInBits,
                   alignInBits, elements, dataLocation, rank, allocated,
                   associated);
    }]>
  ];
  let assemblyFormat = "`<` struct(params) `>`";
  let extraClassDeclaration = [{
    /// Requirements of DIRecursiveTypeAttrInterface.
    /// @{

    /// Get a copy of this type attr but with the recursive ID set to `recId`.
    DIRecursiveTypeAttrInterface withRecId(DistinctAttr recId);

    /// Build a rec-self instance using the provided `recId`.
    static DIRecursiveTypeAttrInterface getRecSelf(DistinctAttr recId);

    /// @}
  }];
}

//===----------------------------------------------------------------------===//
// DIDerivedTypeAttr
//===----------------------------------------------------------------------===//

def LLVM_DIDerivedTypeAttr : LLVM_Attr<"DIDerivedType", "di_derived_type",
                                       /*traits=*/[], "DITypeAttr"> {
  let parameters = (ins
    LLVM_DITagParameter:$tag,
    OptionalParameter<"StringAttr">:$name,
    OptionalParameter<"DITypeAttr">:$baseType,
    OptionalParameter<"uint64_t">:$sizeInBits,
    OptionalParameter<"uint32_t">:$alignInBits,
    OptionalParameter<"uint64_t">:$offsetInBits,
    OptionalParameter<"std::optional<unsigned>">:$dwarfAddressSpace,
    OptionalParameter<"DINodeAttr">:$extraData
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DIFileAttr
//===----------------------------------------------------------------------===//

def LLVM_DIFileAttr : LLVM_Attr<"DIFile", "di_file", /*traits=*/[], "DIScopeAttr"> {
  let parameters = (ins "StringAttr":$name, "StringAttr":$directory);
  let builders = [AttrBuilder<(ins "StringRef":$name, "StringRef":$directory), [{
      return $_get($_ctxt, StringAttr::get($_ctxt, name),
                   StringAttr::get($_ctxt, directory));
    }]>
  ];
  let assemblyFormat = "`<` $name `in` $directory `>`";
}

//===----------------------------------------------------------------------===//
// DIGlobalVariableExpressionAttr
//===----------------------------------------------------------------------===//

def LLVM_DIGlobalVariableExpressionAttr
    : LLVM_Attr<"DIGlobalVariableExpression", "di_global_variable_expression"> {
  let parameters = (ins
    "DIGlobalVariableAttr":$var,
    OptionalParameter<"DIExpressionAttr">:$expr
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let constBuilderCall = "$0";
}

//===----------------------------------------------------------------------===//
// DIGlobalVariableAttr
//===----------------------------------------------------------------------===//

def LLVM_DIGlobalVariable : LLVM_Attr<"DIGlobalVariable", "di_global_variable",
                                      /*traits=*/[], "DINodeAttr"> {
  let parameters = (ins
    OptionalParameter<"DIScopeAttr">:$scope,
    OptionalParameter<"StringAttr">:$name,
    OptionalParameter<"StringAttr">:$linkageName,
    "DIFileAttr":$file,
    "unsigned":$line,
    "DITypeAttr":$type,
    OptionalParameter<"bool">:$isLocalToUnit,
    OptionalParameter<"bool">:$isDefined,
    OptionalParameter<"unsigned">:$alignInBits);
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DILexicalBlockAttr
//===----------------------------------------------------------------------===//

def LLVM_DILexicalBlockAttr : LLVM_Attr<"DILexicalBlock", "di_lexical_block",
                                        /*traits=*/[], "DIScopeAttr"> {
  let parameters = (ins
    "DIScopeAttr":$scope,
    OptionalParameter<"DIFileAttr">:$file,
    OptionalParameter<"unsigned">:$line,
    OptionalParameter<"unsigned">:$column
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins
      "DIScopeAttr":$scope, "DIFileAttr":$file, "unsigned":$line,
      "unsigned":$column
    ), [{
      return $_get(scope.getContext(), scope, file, line, column);
    }]>
  ];
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DILexicalBlockFileAttr
//===----------------------------------------------------------------------===//

def LLVM_DILexicalBlockFile : LLVM_Attr<"DILexicalBlockFile", "di_lexical_block_file",
                                        /*traits=*/[], "DIScopeAttr"> {
  let parameters = (ins
    "DIScopeAttr":$scope,
    OptionalParameter<"DIFileAttr">:$file,
    "unsigned":$discriminator
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins
      "DIScopeAttr":$scope, "DIFileAttr":$file, "unsigned":$discriminator
    ), [{
      return $_get(scope.getContext(), scope, file, discriminator);
    }]>
  ];
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DILocalVariableAttr
//===----------------------------------------------------------------------===//

def LLVM_DILocalVariableAttr : LLVM_Attr<"DILocalVariable", "di_local_variable",
                                         /*traits=*/[], "DINodeAttr"> {
  let parameters = (ins
    "DIScopeAttr":$scope,
    OptionalParameter<"StringAttr">:$name,
    OptionalParameter<"DIFileAttr">:$file,
    OptionalParameter<"unsigned">:$line,
    OptionalParameter<"unsigned">:$arg,
    OptionalParameter<"unsigned">:$alignInBits,
    OptionalParameter<"DITypeAttr">:$type,
    OptionalParameter<"DIFlags", "DIFlags::Zero">:$flags
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins
      "DIScopeAttr":$scope, "StringRef":$name, "DIFileAttr":$file,
      "unsigned":$line, "unsigned":$arg, "unsigned":$alignInBits,
      "DITypeAttr":$type, "DIFlags":$flags
    ), [{
      MLIRContext *ctx = scope.getContext();
      return $_get(ctx, scope, StringAttr::get(ctx, name), file, line,
                   arg, alignInBits, type, flags);
    }]>
  ];
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DISubprogramAttr
//===----------------------------------------------------------------------===//

def LLVM_DISubprogramAttr : LLVM_Attr<"DISubprogram", "di_subprogram",
                                      [LLVM_DIRecursiveTypeAttrInterface],
                                      "DIScopeAttr"> {
  let parameters = (ins
    // DIRecursiveTypeAttrInterface specific parameters.
    OptionalParameter<"DistinctAttr">:$recId,
    OptionalParameter<"bool">:$isRecSelf,
    // DISubprogramAttr specific parameters.
    OptionalParameter<"DistinctAttr">:$id,
    OptionalParameter<"DICompileUnitAttr">:$compileUnit,
    OptionalParameter<"DIScopeAttr">:$scope,
    OptionalParameter<"StringAttr">:$name,
    OptionalParameter<"StringAttr">:$linkageName,
    OptionalParameter<"DIFileAttr">:$file,
    OptionalParameter<"unsigned">:$line,
    OptionalParameter<"unsigned">:$scopeLine,
    OptionalParameter<"DISubprogramFlags">:$subprogramFlags,
    OptionalParameter<"DISubroutineTypeAttr">:$type,
    OptionalArrayRefParameter<"DINodeAttr">:$retainedNodes
  );
  let builders = [
    AttrBuilder<(ins
      "DistinctAttr":$id, "DICompileUnitAttr":$compileUnit,
      "DIScopeAttr":$scope, "StringAttr":$name, "StringAttr":$linkageName,
      "DIFileAttr":$file, "unsigned":$line, "unsigned":$scopeLine,
      "DISubprogramFlags":$subprogramFlags, "DISubroutineTypeAttr":$type,
      "ArrayRef<DINodeAttr>":$retainedNodes
    ), [{
      return $_get($_ctxt, /*recId=*/nullptr, /*isRecSelf=*/false, id, compileUnit,
                   scope, name, linkageName, file, line, scopeLine,
                   subprogramFlags, type, retainedNodes);
    }]>
  ];
  let assemblyFormat = "`<` struct(params) `>`";
  let extraClassDeclaration = [{
    /// Requirements of DIRecursiveTypeAttrInterface.
    /// @{

    /// Get a copy of this type attr but with the recursive ID set to `recId`.
    DIRecursiveTypeAttrInterface withRecId(DistinctAttr recId);

    /// Build a rec-self instance using the provided `recId`.
    static DIRecursiveTypeAttrInterface getRecSelf(DistinctAttr recId);

    /// @}
  }];
}

//===----------------------------------------------------------------------===//
// DIModuleAttr
//===----------------------------------------------------------------------===//

def LLVM_DIModuleAttr : LLVM_Attr<"DIModule", "di_module",
                                      /*traits=*/[], "DIScopeAttr"> {
  let parameters = (ins
    OptionalParameter<"DIFileAttr">:$file,
    OptionalParameter<"DIScopeAttr">:$scope,
    OptionalParameter<"StringAttr">:$name,
    OptionalParameter<"StringAttr">:$configMacros,
    OptionalParameter<"StringAttr">:$includePath,
    OptionalParameter<"StringAttr">:$apinotes,
    OptionalParameter<"unsigned">:$line,
    OptionalParameter<"bool">:$isDecl
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DINamespaceAttr
//===----------------------------------------------------------------------===//

def LLVM_DINamespaceAttr : LLVM_Attr<"DINamespace", "di_namespace",
                                      /*traits=*/[], "DIScopeAttr"> {
  let parameters = (ins
    OptionalParameter<"StringAttr">:$name,
    OptionalParameter<"DIScopeAttr">:$scope,
    "bool":$exportSymbols
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DIImportedEntityAttr
//===----------------------------------------------------------------------===//

def LLVM_DIImportedEntityAttr : LLVM_Attr<"DIImportedEntity", "di_imported_entity",
                                           /*traits=*/[], "DINodeAttr"> {
  let parameters = (ins
    LLVM_DITagParameter:$tag,
    "DIScopeAttr":$scope,
    "DINodeAttr":$entity,
    OptionalParameter<"DIFileAttr">:$file,
    OptionalParameter<"unsigned">:$line,
    OptionalParameter<"StringAttr">:$name,
    OptionalArrayRefParameter<"DINodeAttr">:$elements
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DISubrangeAttr
//===----------------------------------------------------------------------===//

def LLVM_DISubrangeAttr : LLVM_Attr<"DISubrange", "di_subrange", /*traits=*/[],
                                    "DINodeAttr"> {
  let parameters = (ins
    OptionalParameter<"::mlir::Attribute">:$count,
    OptionalParameter<"::mlir::Attribute">:$lowerBound,
    OptionalParameter<"::mlir::Attribute">:$upperBound,
    OptionalParameter<"::mlir::Attribute">:$stride
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DISubroutineTypeAttr
//===----------------------------------------------------------------------===//

def LLVM_DISubroutineTypeAttr : LLVM_Attr<"DISubroutineType", "di_subroutine_type",
                                          /*traits=*/[], "DITypeAttr"> {
  let parameters = (ins
    LLVM_DICallingConventionParameter:$callingConvention,
    OptionalArrayRefParameter<"DITypeAttr">:$types
  );
  let builders = [
    TypeBuilder<(ins "ArrayRef<DITypeAttr>":$types), [{
      return $_get($_ctxt, /*callingConvention=*/0, types);
    }]>
  ];
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DILabelAttr
//===----------------------------------------------------------------------===//

def LLVM_DILabelAttr : LLVM_Attr<"DILabel", "di_label",
                                 /*traits=*/[], "DINodeAttr"> {
  let parameters = (ins
    "DIScopeAttr":$scope,
    OptionalParameter<"StringAttr">:$name,
    OptionalParameter<"DIFileAttr">:$file,
    OptionalParameter<"unsigned">:$line
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins
      "DIScopeAttr":$scope, "StringRef":$name, "DIFileAttr":$file,
      "unsigned":$line
    ), [{
      MLIRContext *ctx = scope.getContext();
      return $_get(ctx, scope, StringAttr::get(ctx, name), file, line);
    }]>
  ];

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DIStringTypeAttr
//===----------------------------------------------------------------------===//

def LLVM_DIStringTypeAttr : LLVM_Attr<"DIStringType", "di_string_type",
                                     /*traits=*/[], "DITypeAttr"> {
  let parameters = (ins
    LLVM_DITagParameter:$tag,
    "StringAttr":$name,
    OptionalParameter<"uint64_t">:$sizeInBits,
    OptionalParameter<"uint32_t">:$alignInBits,
    OptionalParameter<"DIVariableAttr">:$stringLength,
    OptionalParameter<"DIExpressionAttr">:$stringLengthExp,
    OptionalParameter<"DIExpressionAttr">:$stringLocationExp,
    LLVM_DIEncodingParameter:$encoding
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// MemoryEffectsAttr
//===----------------------------------------------------------------------===//

def LLVM_MemoryEffectsAttr : LLVM_Attr<"MemoryEffects", "memory_effects"> {
  let parameters = (ins
    "ModRefInfo":$other,
    "ModRefInfo":$argMem,
    "ModRefInfo":$inaccessibleMem
  );
  let extraClassDeclaration = [{
    bool isReadWrite();
  }];
  let builders = [
    TypeBuilder<(ins "ArrayRef<ModRefInfo>":$memInfoArgs)>
  ];
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// AliasScopeDomainAttr
//===----------------------------------------------------------------------===//

def LLVM_AliasScopeDomainAttr : LLVM_Attr<"AliasScopeDomain",
                                          "alias_scope_domain"> {
  let parameters = (ins
    "DistinctAttr":$id,
    OptionalParameter<"StringAttr">:$description
  );

  let builders = [
    AttrBuilder<(ins CArg<"StringAttr", "{}">:$description), [{
      return $_get($_ctxt, DistinctAttr::create(UnitAttr::get($_ctxt)), description);
    }]>
  ];

  let summary = "LLVM dialect alias scope domain metadata";

  let description = [{
    Defines a domain that may be associated with an alias scope.

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#noalias-and-alias-scope-metadata
  }];

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// AliasScopeAttr
//===----------------------------------------------------------------------===//

def LLVM_AliasScopeAttr : LLVM_Attr<"AliasScope", "alias_scope"> {
  let parameters = (ins
    "DistinctAttr":$id,
    "AliasScopeDomainAttr":$domain,
    OptionalParameter<"StringAttr">:$description
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins
      "AliasScopeDomainAttr":$domain,
      CArg<"StringAttr", "{}">:$description
    ), [{
      MLIRContext *ctx = domain.getContext();
      return $_get(ctx, DistinctAttr::create(UnitAttr::get(ctx)), domain, description);
    }]>
  ];

  let description = [{
    Defines an alias scope that can be attached to a memory-accessing operation.
    Such scopes can be used in combination with `noalias` metadata to indicate
    that sets of memory-affecting operations in one scope do not alias with
    memory-affecting operations in another scope.

    Example:
    ```mlir
    #domain = #llvm.alias_scope_domain<id = distinct[1]<>, description = "Optional domain description">
    #scope1 = #llvm.alias_scope<id = distinct[2]<>, domain = #domain>
    #scope2 = #llvm.alias_scope<id = distinct[3]<>, domain = #domain, description = "Optional scope description">
    llvm.func @foo(%ptr1 : !llvm.ptr) {
        %c0 = llvm.mlir.constant(0 : i32) : i32
        %c4 = llvm.mlir.constant(4 : i32) : i32
        %1 = llvm.ptrtoint %ptr1 : !llvm.ptr to i32
        %2 = llvm.add %1, %c1 : i32
        %ptr2 = llvm.inttoptr %2 : i32 to !llvm.ptr
        llvm.store %c0, %ptr1 { alias_scopes = [#scope1], llvm.noalias = [#scope2] } : i32, !llvm.ptr
        llvm.store %c4, %ptr2 { alias_scopes = [#scope2], llvm.noalias = [#scope1] } : i32, !llvm.ptr
        llvm.return
    }
    ```

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#noalias-and-alias-scope-metadata
  }];

  let summary = "LLVM dialect alias scope";

  let assemblyFormat = "`<` struct(params) `>`";
}

def LLVM_AliasScopeArrayAttr
    : TypedArrayAttrBase<LLVM_AliasScopeAttr,
                         LLVM_AliasScopeAttr.summary # " array"> {
  let constBuilderCall = ?;
}

//===----------------------------------------------------------------------===//
// AccessGroupAttr
//===----------------------------------------------------------------------===//

def LLVM_AccessGroupAttr : LLVM_Attr<"AccessGroup", "access_group"> {

  let parameters = (ins "DistinctAttr":$id);

  let builders = [
    AttrBuilder<(ins), [{
      return $_get($_ctxt, DistinctAttr::create(UnitAttr::get($_ctxt)));
    }]>
  ];

  let summary = "LLVM dialect access group metadata";

  let description = [{
    Defines an access group metadata that can be set on any instruction
    that potentially accesses memory via the `AccessGroupOpInterface` or on
    branch instructions in the loop latch block via the `parallelAccesses`
    parameter of `LoopAnnotationAttr`.

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#llvm-access-group-metadata
  }];

  let assemblyFormat = "`<` struct(params) `>`";
}

def LLVM_AccessGroupArrayAttr
    : TypedArrayAttrBase<LLVM_AccessGroupAttr,
                         LLVM_AccessGroupAttr.summary # " array"> {
  let constBuilderCall = ?;
}

//===----------------------------------------------------------------------===//
// TBAARootAttr
//===----------------------------------------------------------------------===//

def LLVM_TBAARootAttr : LLVM_Attr<"TBAARoot", "tbaa_root", [], "TBAANodeAttr"> {
  let parameters = (ins OptionalParameter<"StringAttr">:$id);

  let summary = "LLVM dialect TBAA root metadata";
  let description = [{
    Defines a TBAA root node.

    Example:
    ```mlir
    #cpp_root = #llvm.tbaa_root<identity = "Simple C/C++ TBAA">
    #other_root = #llvm.tbaa_root
    ```

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#tbaa-metadata
  }];

  let assemblyFormat = "(`<` struct(params)^ `>`)?";
}

//===----------------------------------------------------------------------===//
// TBAATypeDescriptorAttr
//===----------------------------------------------------------------------===//

def LLVM_TBAAMemberAttr : LLVM_Attr<"TBAAMember", "tbaa_member"> {
  let parameters = (ins
    "TBAANodeAttr":$typeDesc,
    "int64_t":$offset
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "TBAANodeAttr":$typeDesc,
                                        "int64_t":$offset), [{
      return $_get(typeDesc.getContext(), typeDesc, offset);
    }]>
  ];

  let assemblyFormat = "`<` params `>`";
}

def LLVM_TBAAMemberAttrArray : ArrayRefParameter<"TBAAMemberAttr"> {
  let printer = [{
    $_printer << '{';
    llvm::interleaveComma($_self, $_printer, [&](TBAAMemberAttr attr) {
        $_printer.printStrippedAttrOrType(attr);
    });
    $_printer << '}';
  }];

  let parser = [{
    [&]() -> FailureOr<SmallVector<TBAAMemberAttr>> {
        using Result = SmallVector<TBAAMemberAttr>;
        if ($_parser.parseLBrace())
            return failure();
        FailureOr<Result> result = FieldParser<Result>::parse($_parser);
        if (failed(result))
            return failure();
        if ($_parser.parseRBrace())
            return failure();
        return result;
    }()
  }];
}

def LLVM_TBAATypeDescriptorAttr : LLVM_Attr<"TBAATypeDescriptor",
    "tbaa_type_desc", [], "TBAANodeAttr"> {
  let parameters = (ins
    StringRefParameter<>:$id,
    LLVM_TBAAMemberAttrArray:$members
  );

  let summary = "LLVM dialect TBAA type metadata";

  let description = [{
    Defines a TBAA node describing a type.

    Example:
    ```mlir
    #tbaa_root = #llvm.tbaa_root<identity = "Simple C/C++ TBAA">
    #tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
    #tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "long long", members = {<#tbaa_root, 0>}>
    #tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "agg2_t", members = {<#tbaa_type_desc2, 0>, <#tbaa_type_desc2, 8>}>
    #tbaa_type_desc4 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc1, 0>}>
    #tbaa_type_desc5 = #llvm.tbaa_type_desc<id = "agg1_t", members = {<#tbaa_type_desc4, 0>, <#tbaa_type_desc4, 4>}>
    ```

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#tbaa-metadata
  }];

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// TBAATagAttr
//===----------------------------------------------------------------------===//

def LLVM_TBAATagAttr : LLVM_Attr<"TBAATag", "tbaa_tag"> {
  let parameters = (ins
    "TBAATypeDescriptorAttr":$base_type,
    "TBAATypeDescriptorAttr":$access_type,
    "int64_t":$offset,
    DefaultValuedParameter<"bool", "false">:$constant
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "TBAATypeDescriptorAttr":$baseType,
                                        "TBAATypeDescriptorAttr":$accessType,
                                        "int64_t":$offset), [{
      return $_get(baseType.getContext(), baseType, accessType, offset,
                    /*constant=*/false);
    }]>
  ];

  let summary = "LLVM dialect TBAA tag metadata";

  let description = [{
    Defines a TBAA node describing a memory access.

    Example:
    ```mlir
    #tbaa_root = #llvm.tbaa_root<identity = "Simple C/C++ TBAA">
    #tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
    #tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc1, 0>}>
    #tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "agg1_t", members = {<#tbaa_type_desc4, 0>, <#tbaa_type_desc4, 4>}>
    #tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc2, offset = 0, constant = true>
    ```

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#tbaa-metadata
  }];

  let assemblyFormat = "`<` struct(params) `>`";
}

def LLVM_TBAATagArrayAttr
    : TypedArrayAttrBase<LLVM_TBAATagAttr,
                         LLVM_TBAATagAttr.summary # " array"> {
  let constBuilderCall = ?;
}

//===----------------------------------------------------------------------===//
// ConstantRangeAttr
//===----------------------------------------------------------------------===//
def LLVM_ConstantRangeAttr : LLVM_Attr<"ConstantRange", "constant_range"> {
  let parameters = (ins
    "::llvm::APInt":$lower,
    "::llvm::APInt":$upper
  );
  let summary = "A range of two integers, corresponding to LLVM's ConstantRange";
  let description = [{
    A pair of two integers, mapping to the ConstantRange structure in LLVM IR,
    which is allowed to wrap or be empty.

    The range represented is [Lower, Upper), and is either signed or unsigned
    depending on context.

    `lower` and `upper` must have the same width.

    Syntax:
    ```
    `<` `i`(width($lower)) $lower `,` $upper `>`
  }];

  let builders = [
    AttrBuilder<(ins "uint32_t":$bitWidth, "int64_t":$lower, "int64_t":$upper), [{
      return $_get($_ctxt, ::llvm::APInt(bitWidth, lower), ::llvm::APInt(bitWidth, upper));
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}


//===----------------------------------------------------------------------===//
// VScaleRangeAttr
//===----------------------------------------------------------------------===//

def LLVM_VScaleRangeAttr : LLVM_Attr<"VScaleRange", "vscale_range"> {
  let parameters =  (ins
    "IntegerAttr":$minRange,
    "IntegerAttr":$maxRange);
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// TargetFeaturesAttr
//===----------------------------------------------------------------------===//

def LLVM_TargetFeaturesAttr : LLVM_Attr<"TargetFeatures", "target_features">
{
  let summary = "LLVM target features attribute";

  let description = [{
    Represents the LLVM target features as a list that can be checked within
    passes/rewrites.

    Example:
    ```mlir
    #llvm.target_features<["+sme", "+sve", "+sme-f64f64"]>
    ```

    Then within a pass or rewrite the features active at an op can be queried:

    ```c++
    auto targetFeatures = LLVM::TargetFeaturesAttr::featuresAt(op);

    if (!targetFeatures.contains("+sme-f64f64"))
      return failure();
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"StringAttr">:$features);

  let builders = [
    TypeBuilder<(ins "::llvm::StringRef":$features)>,
    TypeBuilder<(ins "::llvm::ArrayRef<::llvm::StringRef>":$features)>
  ];

  let extraClassDeclaration = [{
    /// Checks if a feature is contained within the features list.
    /// Note: Using a StringAttr allows doing pointer-comparisons.
    bool contains(::mlir::StringAttr feature) const;
    bool contains(::llvm::StringRef feature) const;

    bool nullOrEmpty() const {
      // Checks if this attribute is null, or the features are empty.
      return !bool(*this) || getFeatures().empty();
    }

    /// Returns the list of features as an LLVM-compatible string.
    std::string getFeaturesString() const;

    /// Finds the target features on the parent FunctionOpInterface.
    /// Note: This assumes the attribute name matches the return value of
    /// `getAttributeName()`.
    static TargetFeaturesAttr featuresAt(Operation* op);

    /// Canonical name for this attribute within MLIR.
    static constexpr StringLiteral getAttributeName() {
      return StringLiteral("target_features");
    }
  }];

  let assemblyFormat = "`<` `[` (`]`) : ($features^ `]`)? `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// UndefAttr
//===----------------------------------------------------------------------===//

/// Folded into from LLVM::UndefOp.
def LLVM_UndefAttr : LLVM_Attr<"Undef", "undef">;

//===----------------------------------------------------------------------===//
// PoisonAttr
//===----------------------------------------------------------------------===//

/// Folded into from LLVM::PoisonOp.
def LLVM_PoisonAttr : LLVM_Attr<"Poison", "poison">;

//===----------------------------------------------------------------------===//
// VecTypeHintAttr
//===----------------------------------------------------------------------===//

def LLVM_VecTypeHintAttr : LLVM_Attr<"VecTypeHint", "vec_type_hint"> {
  let summary = "Explicit vectorization compiler hint";
  let description = [{
    A hint to the compiler that indicates most operations used in the function
    are explictly vectorized using a particular vector type. `$hint` is the
    vector or scalar type in particular. `$is_signed` can be used with integer
    types to state whether the type is signed.
  }];
  let parameters = (ins "TypeAttr":$hint,
                        DefaultValuedParameter<"bool", "false">:$is_signed);
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// ZeroAttr
//===----------------------------------------------------------------------===//

/// Folded into from LLVM::ZeroOp.
def LLVM_ZeroAttr : LLVM_Attr<"Zero", "zero">;

//===----------------------------------------------------------------------===//
// TailCallKindAttr
//===----------------------------------------------------------------------===//

def TailCallKindAttr : LLVM_Attr<"TailCallKind", "tailcallkind"> {
  let parameters = (ins "TailCallKind":$tailCallKind);
  let assemblyFormat = "`<` $tailCallKind `>`";
}

//===----------------------------------------------------------------------===//
// WorkgroupAttributionAttr
//===----------------------------------------------------------------------===//

def WorkgroupAttributionAttr
    : LLVM_Attr<"WorkgroupAttribution", "mlir.workgroup_attribution"> {
  let summary = "GPU workgroup attribution information";
  let description = [{
    GPU workgroup attributions are `gpu.func` attributes encoding memory
    allocations in the workgroup address space. These might be encoded as
    `llvm.ptr` function arguments in our dialect, but then type and size
    information would be dropped. This attribute can be attached to `llvm.ptr`
    function arguments encoding GPU workgroup attributions to mark them as
    arguments encoding workgroup attributions and keeping type and size
    information in our dialect.
  }];
  let parameters = (ins "IntegerAttr":$num_elements,
                        "TypeAttr":$element_type);
  let assemblyFormat = "`<` $num_elements `,` $element_type `>`";
}

#endif // LLVMIR_ATTRDEFS
