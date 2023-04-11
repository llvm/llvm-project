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

//===----------------------------------------------------------------------===//
// LLVM dialect type constraints.
//===----------------------------------------------------------------------===//

// LLVM dialect type.
def LLVM_Type : DialectType<LLVM_Dialect,
                            CPred<"::mlir::LLVM::isCompatibleOuterType($_self)">,
                            "LLVM dialect-compatible type">;

// Type constraint accepting LLVM token type.
def LLVM_TokenType : Type<
  CPred<"$_self.isa<::mlir::LLVM::LLVMTokenType>()">,
  "LLVM token type">,
  BuildableType<"::mlir::LLVM::LLVMTokenType::get($_builder.getContext())">;

// Type constraint accepting LLVM primitive types, i.e. all types except void
// and function.
def LLVM_PrimitiveType : Type<
  And<[LLVM_Type.predicate,
       CPred<"!$_self.isa<::mlir::LLVM::LLVMVoidType, "
                         "::mlir::LLVM::LLVMFunctionType>()">]>,
  "primitive LLVM type">;

// Type constraint accepting any LLVM function type.
def LLVM_FunctionType : Type<CPred<"$_self.isa<::mlir::LLVM::LLVMFunctionType>()">,
                         "LLVM function type", "::mlir::LLVM::LLVMFunctionType">;

// Type constraint accepting any LLVM floating point type.
def LLVM_AnyFloat : Type<
  CPred<"::mlir::LLVM::isCompatibleFloatingPointType($_self)">,
  "floating point LLVM type">;

// Type constraint accepting any LLVM pointer type.
def LLVM_AnyPointer : Type<CPred<"$_self.isa<::mlir::LLVM::LLVMPointerType>()">,
                          "LLVM pointer type", "::mlir::LLVM::LLVMPointerType">;

// Type constraint accepting LLVM pointer type with an additional constraint
// on the element type.
class LLVM_PointerTo<Type pointee> : Type<
  And<[LLVM_AnyPointer.predicate,
       Or<[CPred<"$_self.cast<::mlir::LLVM::LLVMPointerType>().isOpaque()">,
           SubstLeaves<
             "$_self",
             "$_self.cast<::mlir::LLVM::LLVMPointerType>().getElementType()",
             pointee.predicate>]>]>,
  "LLVM pointer to " # pointee.summary, "::mlir::LLVM::LLVMPointerType">;

// Type constraints accepting LLVM pointer type to integer of a specific width.
class LLVM_IntPtrBase<int width, int addressSpace = 0> : Type<
  And<[LLVM_PointerTo<I<width>>.predicate,
       CPred<"$_self.cast<::mlir::LLVM::LLVMPointerType>().getAddressSpace()"
             " == " # addressSpace>]>,
  "LLVM pointer to " # I<width>.summary>;

def LLVM_i8Ptr : LLVM_IntPtrBase<8>;

// Type constraint accepting any LLVM structure type.
def LLVM_AnyStruct : Type<CPred<"$_self.isa<::mlir::LLVM::LLVMStructType>()">,
                         "LLVM structure type">;

// Type constraint accepting opaque LLVM structure type.
def LLVM_OpaqueStruct : Type<
  And<[LLVM_AnyStruct.predicate,
       CPred<"$_self.cast<::mlir::LLVM::LLVMStructType>().isOpaque()">]>>;

// Type constraint accepting any LLVM type that can be loaded or stored, i.e. a
// type that has size (not void, function or opaque struct type).
def LLVM_LoadableType : Type<
  Or<[And<[LLVM_PrimitiveType.predicate, Neg<LLVM_OpaqueStruct.predicate>]>,
      LLVM_PointerElementTypeInterface.predicate]>,
  "LLVM type with size">;

// Type constraint accepting any LLVM aggregate type, i.e. structure or array.
def LLVM_AnyAggregate : Type<
  CPred<"$_self.isa<::mlir::LLVM::LLVMStructType, "
                   "::mlir::LLVM::LLVMArrayType>()">,
  "LLVM aggregate type">;

// Type constraint accepting any LLVM non-aggregate type, i.e. not structure or
// array.
def LLVM_AnyNonAggregate : Type<And<[LLVM_Type.predicate,
                                     Neg<LLVM_AnyAggregate.predicate>]>,
                               "LLVM-compatible non-aggregate type">;

// Type constraint accepting any LLVM vector type.
def LLVM_AnyVector : Type<CPred<"::mlir::LLVM::isCompatibleVectorType($_self)">,
                         "LLVM dialect-compatible vector type">;

// Type constraint accepting any LLVM fixed-length vector type.
def LLVM_AnyFixedVector : Type<CPred<
                                "!::mlir::LLVM::isScalableVectorType($_self)">,
                                "LLVM dialect-compatible fixed-length vector type">;

// Type constraint accepting any LLVM scalable vector type.
def LLVM_AnyScalableVector : Type<CPred<
                                "::mlir::LLVM::isScalableVectorType($_self)">,
                                "LLVM dialect-compatible scalable vector type">;

// Type constraint accepting an LLVM vector type with an additional constraint
// on the vector element type.
class LLVM_VectorOf<Type element> : Type<
  And<[LLVM_AnyVector.predicate,
       SubstLeaves<
         "$_self",
         "::mlir::LLVM::getVectorElementType($_self)",
         element.predicate>]>,
  "LLVM dialect-compatible vector of " # element.summary>;

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
  //   - $_var_attr - substituted by a call to a variable attribute matcher;
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
  code setAccessGroupsMetadataCode = [{
    moduleTranslation.setAccessGroupsMetadata(op, inst);
  }];
  code setAliasAnalysisMetadataCode = [{
    moduleTranslation.setAliasScopeMetadata(op, inst);
    moduleTranslation.setTBAAMetadata(op, inst);
  }];
}

// Patterns with code obtaining the LLVM IR type of the given operand or result
// of operation. "$0" is expected to be replaced by the position of the operand
// or result in the operation.
def LLVM_IntrPatterns {
  string operand =
    [{moduleTranslation.convertType(opInst.getOperand($0).getType())}];
  string result =
    [{moduleTranslation.convertType(opInst.getResult($0).getType())}];
  string structResult =
    [{moduleTranslation.convertType(
        opInst.getResult(0).getType().cast<LLVM::LLVMStructType>()
              .getBody()[$0])}];
}

// For every value in the list, substitutes the value in the place of "$0" in
// "pattern" and stores the list of strings as "lst".
class ListIntSubst<string pattern, list<int> values> {
  list<string> lst = !foreach(x, values,
                              !subst("$0", !cast<string>(x), pattern));
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
  dag aliasAttrs = (ins OptionalAttr<AccessGroupArrayAttr>:$access_groups,
                    OptionalAttr<SymbolRefArrayAttr>:$alias_scopes,
                    OptionalAttr<SymbolRefArrayAttr>:$noalias_scopes,
                    OptionalAttr<SymbolRefArrayAttr>:$tbaa);
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
// translation starting from an LLVM IR intrinsic. The "requiresAccessGroup",
// "requiresAliasAnalysis", and "requiresFastmath" flags specify which
// interfaces the intrinsic implements. If the corresponding flags are set, the
// "aliasAttrs" list contains the arguments required by the access group and
// alias analysis interfaces. Derived intrinsics should append the "aliasAttrs"
// to their argument list if they set one of the flags.
class LLVM_IntrOpBase<Dialect dialect, string opName, string enumName,
                      list<int> overloadedResults, list<int> overloadedOperands,
                      list<Trait> traits, int numResults,
                      bit requiresAccessGroup = 0, bit requiresAliasAnalysis = 0,
                      bit requiresFastmath = 0>
    : LLVM_OpBase<dialect, opName, !listconcat(
        !if(!gt(requiresAccessGroup, 0),
            [DeclareOpInterfaceMethods<AccessGroupOpInterface>], []),
        !if(!gt(requiresAliasAnalysis, 0),
            [DeclareOpInterfaceMethods<AliasAnalysisOpInterface>], []),
        !if(!gt(requiresFastmath, 0),
            [DeclareOpInterfaceMethods<FastmathFlagsInterface>], []),
        traits)>,
      LLVM_MemOpPatterns,
      Results<!if(!gt(numResults, 0), (outs LLVM_Type:$res), (outs))> {
  dag aliasAttrs = !con(
        !if(!gt(requiresAccessGroup, 0),
            (ins OptionalAttr<AccessGroupArrayAttr>:$access_groups),
            (ins )),
        !if(!gt(requiresAccessGroup, 0),
            (ins OptionalAttr<SymbolRefArrayAttr>:$alias_scopes,
                 OptionalAttr<SymbolRefArrayAttr>:$noalias_scopes,
                 OptionalAttr<SymbolRefArrayAttr>:$tbaa),
            (ins )));
  string resultPattern = !if(!gt(numResults, 1),
                             LLVM_IntrPatterns.structResult,
                             LLVM_IntrPatterns.result);
  string llvmEnumName = enumName;
  let llvmBuilder = [{
    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::}] # enumName # [{,
        { }] # !interleave(!listconcat(
            ListIntSubst<resultPattern, overloadedResults>.lst,
            ListIntSubst<LLVM_IntrPatterns.operand,
                         overloadedOperands>.lst), ", ") # [{
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    }] # [{
    auto *inst = builder.CreateCall(fn, operands);
    (void) inst;
    }] # !if(!gt(requiresAccessGroup, 0), setAccessGroupsMetadataCode, "")
       # !if(!gt(requiresAliasAnalysis, 0), setAliasAnalysisMetadataCode, "")
       # !if(!gt(numResults, 0), "$res = inst;", "");

  string mlirBuilder = [{
    FailureOr<SmallVector<Value>> mlirOperands =
      moduleImport.convertValues(llvmOperands);
    if (failed(mlirOperands))
      return failure();
    SmallVector<Type> resultTypes =
    }] # !if(!gt(numResults, 0), "{$_resultType};", "{};") # [{
    auto op = $_builder.create<$_qualCppClassName>(
      $_location, resultTypes, *mlirOperands);
    }] # !if(!gt(requiresFastmath, 0),
      "moduleImport.setFastmathFlagsAttr(inst, op);", "")
    # !if(!gt(numResults, 0), "$res = op;", "$_op = op;");
}

// Base class for LLVM intrinsic operations, should not be used directly. Places
// the intrinsic into the LLVM dialect and prefixes its name with "intr.".
class LLVM_IntrOp<string mnem, list<int> overloadedResults,
                  list<int> overloadedOperands, list<Trait> traits,
                  int numResults, bit requiresAccessGroup = 0,
                  bit requiresAliasAnalysis = 0, bit requiresFastmath = 0>
    : LLVM_IntrOpBase<LLVM_Dialect, "intr." # mnem, !subst(".", "_", mnem),
                      overloadedResults, overloadedOperands, traits,
                      numResults, requiresAccessGroup, requiresAliasAnalysis,
                      requiresFastmath>;

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
                            bit requiresAliasAnalysis = 0>
    : LLVM_IntrOp<mnem, [], overloadedOperands, traits, /*numResults=*/0,
                  requiresAccessGroup, requiresAliasAnalysis>;

// Base class for LLVM intrinsic operations returning one result. Places the
// intrinsic into the LLVM dialect and prefixes its name with "intr.". This is
// similar to LLVM_ZeroResultIntrOp but allows one to define Ops returning one
// result, called "res". Additionally, the overloadedResults list should contain
// "0" if the result must be used to resolve overloaded intrinsics, or remain
// empty otherwise.
class LLVM_OneResultIntrOp<string mnem, list<int> overloadedResults = [],
                           list<int> overloadedOperands = [],
                           list<Trait> traits = [],
                           bit requiresFastmath = 0>
    : LLVM_IntrOp<mnem, overloadedResults, overloadedOperands, traits, 1,
                  /*requiresAccessGroup=*/0, /*requiresAliasAnalysis=*/0,
                  requiresFastmath>;

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
    assert(resultType.isa<LLVMVoidType>() &&
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
