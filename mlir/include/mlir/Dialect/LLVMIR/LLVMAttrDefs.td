//===-- LLVMAttrDefs.td - LLVM Attributes definition file --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_ATTRDEFS
#define LLVMIR_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/LLVMIR/LLVMEnums.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

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
// FastmathFlagsAttr
//===----------------------------------------------------------------------===//

def LLVM_FastmathFlagsAttr :
    EnumAttr<LLVM_Dialect, FastmathFlags, "fastmath"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// LinkageAttr
//===----------------------------------------------------------------------===//

def LinkageAttr : LLVM_Attr<"Linkage", "linkage"> {
  let parameters = (ins "linkage::Linkage":$linkage);
  let assemblyFormat = "`<` $linkage `>`";
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
    OptionalArrayRefParameter<"SymbolRefAttr">:$parallelAccesses
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DebugInfo Attributes
//===----------------------------------------------------------------------===//

class LLVM_DIParameter<string summary, string default, string parseName,
                       string printName = parseName>
    : AttrOrTypeParameter<"unsigned", "debug info " # summary> {
  let parser = [{ [&]() -> FailureOr<unsigned> {
    SMLoc tagLoc = $_parser.getCurrentLocation();
    StringRef name;
    if ($_parser.parseKeyword(&name))
      return failure();

    if (unsigned tag = llvm::dwarf::get}] # parseName # [{(name))
      return tag;
    return $_parser.emitError(tagLoc)
      << "invalid debug info }] # summary # [{ name: " << name;
  }() }];
  let printer = "$_printer << llvm::dwarf::" # printName # "String($_self)";
  let defaultValue = default;
}

def LLVM_DICallingConventionParameter : LLVM_DIParameter<
  "calling convention", /*default=*/"0", "CallingConvention", "Convention"
>;

def LLVM_DIEncodingParameter : LLVM_DIParameter<
  "encoding", /*default=*/"0", "AttributeEncoding"
>;

def LLVM_DILanguageParameter : LLVM_DIParameter<
  "language", /*default=*/"", "Language"
>;

def LLVM_DITagParameter : LLVM_DIParameter<
  "tag", /*default=*/"", "Tag"
>;

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
    LLVM_DILanguageParameter:$sourceLanguage,
    "DIFileAttr":$file,
    "StringAttr":$producer,
    "bool":$isOptimized,
    "DIEmissionKind":$emissionKind
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DICompositeTypeAttr
//===----------------------------------------------------------------------===//

def LLVM_DICompositeTypeAttr : LLVM_Attr<"DICompositeType", "di_composite_type",
                                         /*traits=*/[], "DITypeAttr"> {
  let parameters = (ins
    LLVM_DITagParameter:$tag,
    "StringAttr":$name,
    OptionalParameter<"DIFileAttr">:$file,
    OptionalParameter<"uint32_t">:$line,
    OptionalParameter<"DIScopeAttr">:$scope,
    OptionalParameter<"DITypeAttr">:$baseType,
    OptionalParameter<"DIFlags", "DIFlags::Zero">:$flags,
    OptionalParameter<"uint64_t">:$sizeInBits,
    OptionalParameter<"uint64_t">:$alignInBits,
    OptionalArrayRefParameter<"DINodeAttr">:$elements
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DIDerivedTypeAttr
//===----------------------------------------------------------------------===//

def LLVM_DIDerivedTypeAttr : LLVM_Attr<"DIDerivedType", "di_derived_type",
                                       /*traits=*/[], "DITypeAttr"> {
  let parameters = (ins
    LLVM_DITagParameter:$tag,
    OptionalParameter<"StringAttr">:$name,
    "DITypeAttr":$baseType,
    OptionalParameter<"uint64_t">:$sizeInBits,
    OptionalParameter<"uint32_t">:$alignInBits,
    OptionalParameter<"uint64_t">:$offsetInBits
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
      return $_get(file.getContext(), scope, file, line, column);
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
      return $_get(file.getContext(), scope, file, discriminator);
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
    "StringAttr":$name,
    OptionalParameter<"DIFileAttr">:$file,
    OptionalParameter<"unsigned">:$line,
    OptionalParameter<"unsigned">:$arg,
    OptionalParameter<"unsigned">:$alignInBits,
    OptionalParameter<"DITypeAttr">:$type
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins
      "DIScopeAttr":$scope, "StringRef":$name, "DIFileAttr":$file,
      "unsigned":$line, "unsigned":$arg, "unsigned":$alignInBits,
      "DITypeAttr":$type
    ), [{
      MLIRContext *ctx = file.getContext();
      return $_get(ctx, scope, StringAttr::get(ctx, name), file, line,
                   arg, alignInBits, type);
    }]>
  ];
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DISubprogramAttr
//===----------------------------------------------------------------------===//

def LLVM_DISubprogramAttr : LLVM_Attr<"DISubprogram", "di_subprogram",
                                      /*traits=*/[], "DIScopeAttr"> {
  let parameters = (ins
    OptionalParameter<"DICompileUnitAttr">:$compileUnit,
    "DIScopeAttr":$scope,
    "StringAttr":$name,
    OptionalParameter<"StringAttr">:$linkageName,
    "DIFileAttr":$file,
    OptionalParameter<"unsigned">:$line,
    OptionalParameter<"unsigned">:$scopeLine,
    "DISubprogramFlags":$subprogramFlags,
    OptionalParameter<"DISubroutineTypeAttr">:$type
  );
  let builders = [
    AttrBuilderWithInferredContext<(ins
      "DICompileUnitAttr":$compileUnit, "DIScopeAttr":$scope, "StringRef":$name,
      "StringRef":$linkageName, "DIFileAttr":$file, "unsigned":$line,
      "unsigned":$scopeLine, "DISubprogramFlags":$subprogramFlags,
      "DISubroutineTypeAttr":$type
    ), [{
      MLIRContext *ctx = file.getContext();
      return $_get(ctx, compileUnit, scope, StringAttr::get(ctx, name),
                   StringAttr::get(ctx, linkageName), file, line,
                   scopeLine, subprogramFlags, type);
    }]>
  ];

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DISubrangeAttr
//===----------------------------------------------------------------------===//

def LLVM_DISubrangeAttr : LLVM_Attr<"DISubrange", "di_subrange", /*traits=*/[],
                                    "DINodeAttr"> {
  let parameters = (ins
    OptionalParameter<"IntegerAttr">:$count,
    OptionalParameter<"IntegerAttr">:$lowerBound,
    OptionalParameter<"IntegerAttr">:$upperBound,
    OptionalParameter<"IntegerAttr">:$stride
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

#endif // LLVMIR_ATTRDEFS
