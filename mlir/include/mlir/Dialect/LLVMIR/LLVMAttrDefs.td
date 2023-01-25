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
include "mlir/IR/SubElementInterfaces.td"

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
// LoopOptionsAttr
//===----------------------------------------------------------------------===//

def LoopOptionsAttr : LLVM_Attr<"LoopOptions", "loopopts"> {
  let description = [{
    This attributes encapsulates "loop options". It is means to decorate
    branches that are "latches" (loop backedges) and maps to the `!llvm.loop`
    metadatas: https://llvm.org/docs/LangRef.html#llvm-loop
    It store the options as a pair <enum,int64_t> in a sorted array and expose
    APIs to retrieve the value for each option with a stronger type (bool for
    example).
  }];

  let parameters = (ins
    ArrayRefParameter<"std::pair<LoopOptionCase, int64_t>", "">:$options
  );

  let extraClassDeclaration = [{
    using OptionValuePair = std::pair<LoopOptionCase, int64_t>;
    using OptionsArray = ArrayRef<std::pair<LoopOptionCase, int64_t>>;
    std::optional<bool> disableUnroll();
    std::optional<bool> disableLICM();
    std::optional<int64_t> interleaveCount();
  }];

  let builders = [
    /// Build the LoopOptions Attribute from a sorted array of individual options.
    AttrBuilder<(ins "ArrayRef<std::pair<LoopOptionCase, int64_t>>":$sortedOptions)>,
    AttrBuilder<(ins "LoopOptionsAttrBuilder &":$optionBuilders)>
  ];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
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
// DIVoidResultTypeAttr
//===----------------------------------------------------------------------===//

def LLVM_DIVoidResultTypeAttr : LLVM_Attr<"DIVoidResultType", "di_void_result_type",
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

def LLVM_DICompileUnitAttr : LLVM_Attr<"DICompileUnit", "di_compile_unit", [
    SubElementAttrInterface
  ], "DIScopeAttr"> {
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

def LLVM_DICompositeTypeAttr : LLVM_Attr<"DICompositeType", "di_composite_type", [
    SubElementAttrInterface
  ], "DITypeAttr"> {
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

def LLVM_DIDerivedTypeAttr : LLVM_Attr<"DIDerivedType", "di_derived_type", [
    SubElementAttrInterface
  ], "DITypeAttr"> {
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

def LLVM_DILexicalBlockAttr : LLVM_Attr<"DILexicalBlock", "di_lexical_block", [
    SubElementAttrInterface
  ], "DIScopeAttr"> {
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

def LLVM_DILexicalBlockFile : LLVM_Attr<"DILexicalBlockFile", "di_lexical_block_file", [
    SubElementAttrInterface
  ], "DIScopeAttr"> {
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

def LLVM_DILocalVariableAttr : LLVM_Attr<"DILocalVariable", "di_local_variable", [
    SubElementAttrInterface
  ], "DINodeAttr"> {
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

def LLVM_DISubprogramAttr : LLVM_Attr<"DISubprogram", "di_subprogram", [
    SubElementAttrInterface
  ], "DIScopeAttr"> {
  let parameters = (ins
    "DICompileUnitAttr":$compileUnit,
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

def LLVM_DISubroutineTypeAttr : LLVM_Attr<"DISubroutineType", "di_subroutine_type", [
    SubElementAttrInterface
  ], "DITypeAttr"> {
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
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// MemoryEffectsAttr
//===----------------------------------------------------------------------===//

def LLVM_MemoryEffectsAttr : LLVM_Attr<"MemoryEffects", "memory_effects", [
    SubElementAttrInterface
  ]> {
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
