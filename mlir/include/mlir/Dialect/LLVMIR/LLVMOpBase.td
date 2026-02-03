//===-- LLVMOpBase.td - LLVM IR dialect shared definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains shared definitions for the LLVM IR dialect and its
// subdialects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_OP_BASE
#define LLVMIR_OP_BASE

include "mlir/Dialect/LLVMIR/LLVMAttrDefs.td"
include "mlir/Dialect/LLVMIR/LLVMInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

//===----------------------------------------------------------------------===//
// LLVM dialect type constraints.
//===----------------------------------------------------------------------===//

// LLVM dialect type.
def LLVM_Type : DialectType<LLVM_Dialect,
                            CPred<"::mlir::LLVM::isCompatibleOuterType($_self)">,
                            "LLVM dialect-compatible type">;

// Type constraint accepting LLVM token type.
def LLVM_TokenType : Type<
  CPred<"::llvm::isa<::mlir::LLVM::LLVMTokenType>($_self)">,
  "LLVM token type">,
  BuildableType<"::mlir::LLVM::LLVMTokenType::get($_builder.getContext())">;

// Type constraint accepting LLVM primitive types, i.e. all types except void
// and function.
def LLVM_PrimitiveType : Type<
  And<[LLVM_Type.predicate,
       CPred<"!::llvm::isa<::mlir::LLVM::LLVMVoidType, "
                         "::mlir::LLVM::LLVMFunctionType>($_self)">]>,
  "primitive LLVM type">;

// Type constraint accepting any LLVM function type.
def LLVM_FunctionType : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMFunctionType>($_self)">,
                         "LLVM function type", "::mlir::LLVM::LLVMFunctionType">;

// Type constraint accepting any LLVM floating point type.
def LLVM_AnyFloat : Type<
  CPred<"::mlir::LLVM::isCompatibleFloatingPointType($_self)">,
  "floating point LLVM type">;

// Type constraint accepting any LLVM pointer type.
def LLVM_AnyPointer : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">,
                          "LLVM pointer type", "::mlir::LLVM::LLVMPointerType">;

// Pointer in a given address space.
class LLVM_PointerInAddressSpace<int addressSpace> : Type<
  And<[LLVM_AnyPointer.predicate,
      CPred<
        "::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self).getAddressSpace() == "
        # addressSpace>]>,
  "LLVM pointer in address space " # addressSpace,
  "::mlir::LLVM::LLVMPointerType"> {
  let builderCall = "$_builder.getType<::mlir::LLVM::LLVMPointerType>("
    # addressSpace # ")";
}

// Type constraint accepting an LLVM pointer type in address space 0.
def LLVM_DefaultPointer : LLVM_PointerInAddressSpace<0>;

// Type constraint accepting any LLVM structure type.
def LLVM_AnyStruct : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMStructType>($_self)">,
                         "LLVM structure type">;

// Type constraint accepting opaque LLVM structure type.
def LLVM_OpaqueStruct : Type<
  And<[LLVM_AnyStruct.predicate,
       CPred<"::llvm::cast<::mlir::LLVM::LLVMStructType>($_self).isOpaque()">]>>;

// Type constraint accepting any LLVM target extension type.
def LLVM_AnyTargetExt : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMTargetExtType>($_self)">,
                            "LLVM target extension type">;

// Type constraint accepting LLVM target extension types with no support for
// memory operations such as alloca, load and store.
def LLVM_NonLoadableTargetExtType : Type<
  And<[LLVM_AnyTargetExt.predicate,
        CPred<"!::llvm::cast<::mlir::LLVM::LLVMTargetExtType>($_self).supportsMemOps()">]
        >>;

// Type constraint accepting any LLVM type that can be loaded or stored, i.e. a
// type that has size (not void, function, opaque struct type or target
// extension type which does not support memory operations).
def LLVM_LoadableType : Type<
  Or<[CPred<"mlir::LLVM::isLoadableType($_self)">,
      LLVM_PointerElementTypeInterface.predicate]>,
  "LLVM type with size">;

// Type constraint accepting any LLVM aggregate type, i.e. structure or array.
def LLVM_AnyAggregate : Type<
  CPred<"::llvm::isa<::mlir::LLVM::LLVMStructType, "
                   "::mlir::LLVM::LLVMArrayType>($_self)">,
  "LLVM aggregate type">;

// Type constraint accepting any LLVM non-aggregate type, i.e. not structure or
// array.
def LLVM_AnyNonAggregate : Type<And<[LLVM_Type.predicate,
                                     Neg<LLVM_AnyAggregate.predicate>]>,
                               "LLVM-compatible non-aggregate type">;

// Type constraint accepting any LLVM vector type.
def LLVM_AnyVector : Type<CPred<"::mlir::LLVM::isCompatibleVectorType($_self)">,
                         "LLVM dialect-compatible vector type",
                         "::mlir::VectorType">;

// Type constraint accepting any LLVM fixed-length vector type.
def LLVM_AnyFixedVector : Type<CPred<
                                "!::mlir::LLVM::isScalableVectorType($_self)">,
                                "LLVM dialect-compatible fixed-length vector type",
                                "::mlir::VectorType">;

// Type constraint accepting any LLVM scalable vector type.
def LLVM_AnyScalableVector : Type<CPred<
                                "::mlir::LLVM::isScalableVectorType($_self)">,
                                "LLVM dialect-compatible scalable vector type",
                                "::mlir::VectorType">;

// Type constraint accepting an LLVM vector type with an additional constraint
// on the vector element type.
class LLVM_VectorOf<Type element> : Type<
  And<[LLVM_AnyVector.predicate,
       SubstLeaves<
         "$_self",
         "::llvm::cast<::mlir::VectorType>($_self).getElementType()",
         element.predicate>]>,
  "LLVM dialect-compatible vector of " # element.summary,
  "::mlir::VectorType">;

// Type constraint accepting a constrained type, or a vector of such types.
class LLVM_ScalarOrVectorOf<Type element> :
    AnyTypeOf<[element, LLVM_VectorOf<element>]>;

// Base class for LLVM operations. Defines the interface to the llvm::IRBuilder
// used to translate to proper LLVM IR and the interface to the mlir::OpBuilder
// used to import from LLVM IR.
class LLVM_OpBase<Dialect dialect, string mnemonic, list<Trait> traits = []> :
    Op<dialect, mnemonic, traits> {
  // A pattern for constructing the LLVM IR Instruction (or other Value) that
  // corresponds to this op.  This pattern can use `builder` to refer to an
  // `llvm::IRBuilder<>` instance, $-names of arguments and results and the
  // following special variable names:
  //   - $_resultType - substituted with the LLVM IR type of the result;
  //   - $_numOperands - substituted with the number of operands (including
  //                     the variadic ones);
  //   - $_hasResult - substituted with a check that a variadic-result op does
  //                   have a result (LLVM ops can have 0 or 1 result);
  //   - $_location - mlir::Location object of the instruction.
  // Additionally, `$$` can be used to produce the dollar character.
  string llvmBuilder = "";

  // A builder to construct the MLIR LLVM dialect operation given the matching
  // LLVM IR instruction `inst` and its operands `llvmOperands`. The
  // following $-variables exist:
  //   - $name - substituted by the remapped `inst` operand value at the index
  //             of the MLIR operation argument with the given name, or if the
  //             name matches the result name, by a reference to store the
  //             result of the newly created MLIR operation to;
  //   - $_op - substituted by a reference to store the newly created MLIR
  //            operation (only for MLIR operations that return no result);
  //   - $_int_attr - substituted by a call to an integer attribute matcher;
  //   - $_float_attr - substituted by a call to a float attribute matcher;
  //   - $_var_attr - substituted by a call to a variable attribute matcher;
  //   - $_label_attr - substituted by a call to a label attribute matcher;
  //   - $_roundingMode_attr - substituted by a call to a rounding mode
  //     attribute matcher;
  //   - $_fpExceptionBehavior_attr - substituted by a call to a FP exception
  //     behavior attribute matcher;
  //   - $_resultType - substituted with the MLIR result type;
  //   - $_location - substituted with the MLIR location;
  //   - $_builder - substituted with the MLIR builder;
  //   - $_qualCppClassName - substitiuted with the MLIR operation class name.
  // Always either store a reference to the result of the newly created
  // operation, or to the operation itself if it does not return a result.
  // Additionally, `$$` can be used to produce the dollar character.
  string mlirBuilder = "";

  // An array that specifies a mapping from MLIR argument indices to LLVM IR
  // operand indices. The mapping is necessary since argument and operand
  // indices do not always match. If not defined, the array is set to the
  // identity permutation. An operation may define any custom index permutation
  // and set a specific argument index to -1 if it does not map to an LLVM IR
  // operand.
  list<int> llvmArgIndices = [];
}

//===----------------------------------------------------------------------===//
// Patterns for LLVM dialect operations.
//===----------------------------------------------------------------------===//

// Patterns with code to set flags and metadata of memory operations after their
// translation to LLVM IR instructions. Operations may use the patterns to
// implement their "llvmBuilder". The patterns assume the `op` and `inst`
// variables exist and refer to the original MLIR operation and the translated
// LLVM IR instruction, respectively.
class LLVM_MemOpPatterns {
  code setAlignmentCode = [{
    if ($alignment.has_value()) {
      auto align = *$alignment;
      if (align != 0)
        inst->setAlignment(llvm::Align(align));
    }
  }];
  code setVolatileCode = [{
    inst->setVolatile($volatile_);
  }];
  code setSyncScopeCode = [{
    if ($syncscope.has_value()) {
      llvm::LLVMContext &llvmContext = builder.getContext();
      inst->setSyncScopeID(llvmContext.getOrInsertSyncScopeID(*$syncscope));
    }
  }];
  code setOrderingCode = [{
    inst->setAtomic(convertAtomicOrderingToLLVM($ordering));
  }];
  code setNonTemporalMetadataCode = [{
    if ($nontemporal) {
      llvm::MDNode *metadata = llvm::MDNode::get(
          inst->getContext(), llvm::ConstantAsMetadata::get(
              builder.getInt32(1)));
      inst->setMetadata(llvm::LLVMContext::MD_nontemporal, metadata);
    }
  }];
  code setInvariantGroupCode = [{
    if ($invariantGroup) {
      llvm::MDNode *metadata = llvm::MDNode::get(inst->getContext(), {});
      inst->setMetadata(llvm::LLVMContext::MD_invariant_group, metadata);
    }
  }];
  code setAccessGroupsMetadataCode = [{
    moduleTranslation.setAccessGroupsMetadata(op, inst);
  }];
  code setAliasAnalysisMetadataCode = [{
    moduleTranslation.setAliasScopeMetadata(op, inst);
    moduleTranslation.setTBAAMetadata(op, inst);
  }];
}

//===----------------------------------------------------------------------===//
// Base classes for LLVM dialect operations.
//===----------------------------------------------------------------------===//

// Base class for LLVM operations. All operations get an "llvm." prefix in
// their name automatically and should either have zero or one result.
class LLVM_Op<string mnemonic, list<Trait> traits = []> :
    LLVM_OpBase<LLVM_Dialect, mnemonic, traits>;

// Base class for LLVM memory access operations that implement the access group
// and alias analysis interfaces. The "aliasAttrs" list contains the arguments
// required by the access group and alias analysis interfaces. Derived
// operations should append the "aliasAttrs" to their argument list.
class LLVM_MemAccessOpBase<string mnemonic, list<Trait> traits = []> :
    LLVM_Op<mnemonic, !listconcat([
      DeclareOpInterfaceMethods<AccessGroupOpInterface>,
      DeclareOpInterfaceMethods<AliasAnalysisOpInterface>], traits)>,
    LLVM_MemOpPatterns {
  dag aliasAttrs = (ins OptionalAttr<LLVM_AccessGroupArrayAttr>:$access_groups,
                    OptionalAttr<LLVM_AliasScopeArrayAttr>:$alias_scopes,
                    OptionalAttr<LLVM_AliasScopeArrayAttr>:$noalias_scopes,
                    OptionalAttr<LLVM_TBAATagArrayAttr>:$tbaa);
}

// Base class for LLVM intrinsics operation. It is similar to LLVM_Op, but
// provides the "llvmBuilder" field for constructing the intrinsic.
// The builder relies on the contents of "overloadedResults" and
// "overloadedOperands" lists that contain the positions of intrinsic results
// and operands that are overloadable in the LLVM sense, that is their types
// must be passed in during the construction of the intrinsic declaration to
// differentiate between differently-typed versions of the intrinsic.
// If the intrinsic has multiple results, this will eventually be packed into a
// single struct result. In this case, the types of any overloaded results need
// to be accessed via the LLVMStructType, instead of directly via the result.
// "opName" contains the name of the operation to be associated with the
// intrinsic and "enumName" contains the name of the intrinsic as appears in
// `llvm::Intrinsic` enum; one usually wants these to be related. Additionally,
// the base class also defines the "mlirBuilder" field to support the inverse
// translation starting from an LLVM IR intrinsic.
//
// The flags "requiresAccessGroup", "requiresAliasAnalysis",
// "requiresFastmath", and "requiresArgAndResultAttrs" indicate which
// interfaces the intrinsic implements. When a flag is set, the "baseArgs"
// list includes the arguments required by the corresponding interface.
// Derived intrinsics must append "baseArgs" to their argument list if they
// enable any of these flags.
//
// LLVM `immargs` can be represented as MLIR attributes by providing both
// the `immArgPositions` and `immArgAttrNames` lists. These two lists should
// have equal length, with `immArgPositions` containing the argument
// positions on the LLVM IR attribute that are `immargs`, and
// `immArgAttrNames` mapping these to corresponding MLIR attributes.
class LLVM_IntrOpBase<Dialect dialect, string opName, string enumName,
                      list<int> overloadedResults, list<int> overloadedOperands,
                      list<Trait> traits, int numResults,
                      bit requiresAccessGroup = 0, bit requiresAliasAnalysis = 0,
                      bit requiresFastmath = 0, bit requiresArgAndResultAttrs = 0,
                      bit requiresOpBundles = 0,
                      list<int> immArgPositions = [],
                      list<string> immArgAttrNames = []>
    : LLVM_OpBase<dialect, opName, !listconcat(
        !if(!gt(requiresAccessGroup, 0),
            [DeclareOpInterfaceMethods<AccessGroupOpInterface>], []),
        !if(!gt(requiresAliasAnalysis, 0),
            [DeclareOpInterfaceMethods<AliasAnalysisOpInterface>], []),
        !if(!gt(requiresFastmath, 0),
            [DeclareOpInterfaceMethods<FastmathFlagsInterface>], []),
        !if(!gt(requiresArgAndResultAttrs, 0),
            [DeclareOpInterfaceMethods<ArgAndResultAttrsOpInterface>], []),
        traits)>,
      LLVM_MemOpPatterns,
      Results<!if(!gt(numResults, 0), (outs LLVM_Type:$res), (outs))> {
  dag baseArgs = !con(
        !if(!gt(requiresAccessGroup, 0),
            (ins OptionalAttr<LLVM_AccessGroupArrayAttr>:$access_groups),
            (ins )),
        !if(!gt(requiresAliasAnalysis, 0),
            (ins OptionalAttr<LLVM_AliasScopeArrayAttr>:$alias_scopes,
                 OptionalAttr<LLVM_AliasScopeArrayAttr>:$noalias_scopes,
                 OptionalAttr<LLVM_TBAATagArrayAttr>:$tbaa),
            (ins )),
        !if(!gt(requiresArgAndResultAttrs, 0),
            (ins OptionalAttr<DictArrayAttr>:$arg_attrs,
                 OptionalAttr<DictArrayAttr>:$res_attrs),
            (ins )),
        !if(!gt(requiresOpBundles, 0),
            (ins VariadicOfVariadic<LLVM_Type,
                  "op_bundle_sizes">:$op_bundle_operands,
                 DenseI32ArrayAttr:$op_bundle_sizes,
                 OptionalAttr<ArrayAttr>:$op_bundle_tags),
            (ins )));
  string llvmEnumName = enumName;
  string overloadedResultsCpp =  "{" # !interleave(overloadedResults, ", ") # "}";
  string overloadedOperandsCpp =  "{" # !interleave(overloadedOperands, ", ") # "}";
  string immArgPositionsCpp = "{" # !interleave(immArgPositions, ", ") # "}";
  string immArgAttrNamesCpp = "{" # !interleave(!foreach(name, immArgAttrNames,
    "StringLiteral(\"" # name # "\")"), ", ") # "}";
  string baseLlvmBuilder = [{
    auto *inst = LLVM::detail::createIntrinsicCall(
      builder, moduleTranslation, &opInst, llvm::Intrinsic::}] # !interleave([
        enumName, "" # numResults, overloadedResultsCpp, overloadedOperandsCpp,
        immArgPositionsCpp, immArgAttrNamesCpp], ",") # [{);
    (void) inst;
    }];
  string baseLlvmBuilderArgAndResultAttrs = [{
    if (failed(moduleTranslation.convertArgAndResultAttrs(
        op,
        inst,
        }] # immArgPositionsCpp # [{))) {
      return failure();
    }
  }];
  string baseLlvmBuilderCoda = !if(!gt(numResults, 0), "$res = inst;", "");
  let llvmBuilder = baseLlvmBuilder
      # !if(!gt(requiresAccessGroup, 0),
        setAccessGroupsMetadataCode, "")
      # !if(!gt(requiresAliasAnalysis, 0),
        setAliasAnalysisMetadataCode, "")
      # !if(!gt(requiresArgAndResultAttrs, 0),
        baseLlvmBuilderArgAndResultAttrs, "")
      # baseLlvmBuilderCoda;

  string baseMlirBuilder = [{
    SmallVector<Value> mlirOperands;
    SmallVector<NamedAttribute> mlirAttrs;
    if (failed(moduleImport.convertIntrinsicArguments(
        llvmOperands,
        llvmOpBundles,
        }] # !if(!gt(requiresOpBundles, 0), "true", "false") # [{,
        }] # immArgPositionsCpp # [{,
        }] # immArgAttrNamesCpp # [{,
        mlirOperands,
        mlirAttrs))) {
      return failure();
    }
    SmallVector<Type> resultTypes =
    }] # !if(!gt(numResults, 0), "{$_resultType};", "{};") # [{
    auto op = $_qualCppClassName::create($_builder,
      $_location, resultTypes, mlirOperands, mlirAttrs);
    }];
  string baseMlirBuilderArgAndResultAttrs = [{
    moduleImport.convertArgAndResultAttrs(
      inst, op, }] # immArgPositionsCpp # [{);
    }];
  string baseMlirBuilderCoda = !if(!gt(numResults, 0), "$res = op;", "$_op = op;");
  let mlirBuilder = baseMlirBuilder
    # !if(!gt(requiresFastmath, 0),
      "moduleImport.setFastmathFlagsAttr(inst, op);", "")
    # !if(!gt(requiresArgAndResultAttrs, 0),
      baseMlirBuilderArgAndResultAttrs, "")
    # baseMlirBuilderCoda;

  // Code for handling a `range` attribute that holds the constant range of the
  // intrinsic's result (if one is specified at the call site). This is intended
  // for GPU IDs and other calls where range() is meaningful. It expects
  // an optional LLVM_ConstantRangeAttr named `range` to be present on the
  // operation. These are included to abstract out common code in several
  // dialects.
  string setRangeRetAttrCode = [{
    if ($range) {
      inst->addRangeRetAttr(::llvm::ConstantRange(
        $range->getLower(), $range->getUpper()));
    }
  }];
  string importRangeRetAttrCode = [{
    // Note: we don't want to look in to the declaration here.
    auto rangeAttr = inst->getAttributes().getRetAttr(::llvm::Attribute::Range);
    if (rangeAttr.isValid()) {
      const ::llvm::ConstantRange& value = rangeAttr.getValueAsConstantRange();
      op.setRangeAttr(::mlir::LLVM::ConstantRangeAttr::get($_builder.getContext(), value.getLower(), value.getUpper()));
    }
  }];
}

// Base class for LLVM intrinsic operations, should not be used directly. Places
// the intrinsic into the LLVM dialect and prefixes its name with "intr.".
class LLVM_IntrOp<string mnem, list<int> overloadedResults,
                  list<int> overloadedOperands, list<Trait> traits,
                  int numResults, bit requiresAccessGroup = 0,
                  bit requiresAliasAnalysis = 0, bit requiresFastmath = 0,
                  bit requiresArgAndResultAttrs = 0, bit requiresOpBundles = 0,
                  list<int> immArgPositions = [],
                  list<string> immArgAttrNames = []>
    : LLVM_IntrOpBase<LLVM_Dialect, "intr." # mnem, !subst(".", "_", mnem),
                      overloadedResults, overloadedOperands, traits,
                      numResults, requiresAccessGroup, requiresAliasAnalysis,
                      requiresFastmath, requiresArgAndResultAttrs,
                      requiresOpBundles, immArgPositions, immArgAttrNames>;

// Base class for LLVM intrinsic operations returning no results. Places the
// intrinsic into the LLVM dialect and prefixes its name with "intr.".
//
// Sample use: derive an entry from this class and populate the fields.
//
//    def LLVM_Name : LLVM_ZeroResultIntrOp<"name", [0], [Pure]>,
//                    Arguments<(ins LLVM_Type, LLVM_Type)>;
//
// The mnemonic will be prefixed with "llvm.intr.", where the "llvm." part comes
// from the LLVM dialect. The overloadedOperands list contains the indices of
// the operands the type of which will be passed in the LLVM IR intrinsic
// builder. In the example above, the Op has two arguments, but only the first
// one (as indicated by `[0]`) is necessary to resolve the overloaded intrinsic.
// The Op has no results.
class LLVM_ZeroResultIntrOp<string mnem, list<int> overloadedOperands = [],
                            list<Trait> traits = [],
                            bit requiresAccessGroup = 0,
                            bit requiresAliasAnalysis = 0,
                            bit requiresArgAndResultAttrs = 0,
                            bit requiresOpBundles = 0,
                            list<int> immArgPositions = [],
                            list<string> immArgAttrNames = []>
    : LLVM_IntrOp<mnem, [], overloadedOperands, traits, /*numResults=*/0,
                  requiresAccessGroup, requiresAliasAnalysis,
                  /*requiresFastMath=*/0, requiresArgAndResultAttrs,
                  requiresOpBundles, immArgPositions, immArgAttrNames>;

// Base class for LLVM intrinsic operations returning one result. Places the
// intrinsic into the LLVM dialect and prefixes its name with "intr.". This is
// similar to LLVM_ZeroResultIntrOp but allows one to define Ops returning one
// result, called "res". Additionally, the overloadedResults list should contain
// "0" if the result must be used to resolve overloaded intrinsics, or remain
// empty otherwise.
class LLVM_OneResultIntrOp<string mnem, list<int> overloadedResults = [],
                           list<int> overloadedOperands = [],
                           list<Trait> traits = [],
                           bit requiresFastmath = 0,
                           bit requiresArgAndResultAttrs = 0,
                           list<int> immArgPositions = [],
                           list<string> immArgAttrNames = []>
    : LLVM_IntrOp<mnem, overloadedResults, overloadedOperands, traits, 1,
                  /*requiresAccessGroup=*/0, /*requiresAliasAnalysis=*/0,
                  requiresFastmath, requiresArgAndResultAttrs,
                  /*requiresOpBundles=*/0, immArgPositions,
                  immArgAttrNames>;

// Base class for LLVM intrinsic operations returning two results. Places the
// intrinsic into the LLVM dialect and prefixes its name with "intr.". This is
// similar to LLVM_ZeroResultIntrOp but allows one to define Ops returning two
// results. Additionally, the overloadedResults list should contain "0", "1"
// if the result must be used to resolve overloaded intrinsics, or remain
// empty otherwise.
class LLVM_TwoResultIntrOp<string mnem, list<int> overloadedResults = [],
                           list<int> overloadedOperands = [],
                           list<Trait> traits = [],
                           bit requiresFastmath = 0,
                           list<int> immArgPositions = [],
                           list<string> immArgAttrNames = []>
    : LLVM_IntrOp<mnem, overloadedResults, overloadedOperands, traits, 2,
                  /*requiresAccessGroup=*/0, /*requiresAliasAnalysis=*/0,
                  requiresFastmath, /*requiresArgAndResultAttrs=*/0,
                  /*requiresOpBundles=*/0, immArgPositions,
                  immArgAttrNames>;

def LLVM_OneResultOpBuilder :
  OpBuilder<(ins "Type":$resultType, "ValueRange":$operands,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    if (resultType) $_state.addTypes(resultType);
    $_state.addOperands(operands);
    for (auto namedAttr : attributes)
      $_state.addAttribute(namedAttr.getName(), namedAttr.getValue());
  }]>;

def LLVM_ZeroResultOpBuilder :
  OpBuilder<(ins "ValueRange":$operands,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(operands);
    for (auto namedAttr : attributes)
      $_state.addAttribute(namedAttr.getName(), namedAttr.getValue());
  }]>;

// Compatibility builder that takes an instance of wrapped llvm::VoidType
// to indicate no result.
def LLVM_VoidResultTypeOpBuilder :
  OpBuilder<(ins "Type":$resultType, "ValueRange":$operands,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    assert(isCompatibleType(resultType) && "result must be an LLVM type");
    assert(::llvm::isa<LLVMVoidType>(resultType) &&
           "for zero-result operands, only 'void' is accepted as result type");
    build($_builder, $_state, operands, attributes);
  }]>;


// Opaque builder used for terminator operations that contain successors.
def LLVM_TerminatorPassthroughOpBuilder :
  OpBuilder<(ins "ValueRange":$operands, "SuccessorRange":$destinations,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(operands);
    $_state.addSuccessors(destinations);
    $_state.addAttributes(attributes);
  }]>;


#endif  // LLVMIR_OP_BASE
