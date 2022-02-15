//===-- LLVMOps.td - LLVM IR dialect op definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the LLVM IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_OPS
#define LLVMIR_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpsInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def FMFnnan     : BitEnumAttrCaseBit<"nnan", 0>;
def FMFninf     : BitEnumAttrCaseBit<"ninf", 1>;
def FMFnsz      : BitEnumAttrCaseBit<"nsz", 2>;
def FMFarcp     : BitEnumAttrCaseBit<"arcp", 3>;
def FMFcontract : BitEnumAttrCaseBit<"contract", 4>;
def FMFafn      : BitEnumAttrCaseBit<"afn", 5>;
def FMFreassoc  : BitEnumAttrCaseBit<"reassoc", 6>;
def FMFfast     : BitEnumAttrCaseBit<"fast", 7>;

def FastmathFlags_DoNotUse : BitEnumAttr<
    "FastmathFlags",
    "LLVM fastmath flags",
    [FMFnnan, FMFninf, FMFnsz, FMFarcp, FMFcontract, FMFafn, FMFreassoc, FMFfast
    ]> {
  let cppNamespace = "::mlir::LLVM";
}

def LLVM_FMFAttr : DialectAttr<
    LLVM_Dialect,
    CPred<"$_self.isa<::mlir::LLVM::FMFAttr>()">,
    "LLVM fastmath flags"> {
  let storageType = "::mlir::LLVM::FMFAttr";
  let returnType = "::mlir::LLVM::FastmathFlags";
  let convertFromStorage = "$_self.getFlags()";
  let constBuilderCall =
          "::mlir::LLVM::FMFAttr::get($_builder.getContext(), $0)";
}

def LOptDisableUnroll : I32EnumAttrCase<"disable_unroll", 1>;
def LOptDisableLICM : I32EnumAttrCase<"disable_licm", 2>;
def LOptInterleaveCount : I32EnumAttrCase<"interleave_count", 3>;
def LOptDisablePipeline : I32EnumAttrCase<"disable_pipeline", 4>;
def LOptPipelineInitiationInterval : I32EnumAttrCase<"pipeline_initiation_interval", 5>;

def LoopOptionCase : I32EnumAttr<
    "LoopOptionCase",
    "LLVM loop option",
    [LOptDisableUnroll, LOptDisableLICM, LOptInterleaveCount,
     LOptDisablePipeline, LOptPipelineInitiationInterval
    ]> {
  let cppNamespace = "::mlir::LLVM";
}

class LLVM_Builder<string builder> {
  string llvmBuilder = builder;
}

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

// Base class for LLVM terminator operations.  All terminator operations have
// zero results and an optional list of successors.
class LLVM_TerminatorOp<string mnemonic, list<Trait> traits = []> :
    LLVM_Op<mnemonic, !listconcat(traits, [Terminator])>;

// Class for arithmetic binary operations.
class LLVM_ArithmeticOpBase<Type type, string mnemonic,
                            string builderFunc, list<Trait> traits = []> :
    LLVM_Op<mnemonic,
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)>,
    LLVM_Builder<"$res = builder." # builderFunc # "($lhs, $rhs);"> {
  dag commonArgs = (ins LLVM_ScalarOrVectorOf<type>:$lhs,
                    LLVM_ScalarOrVectorOf<type>:$rhs);
  let results = (outs LLVM_ScalarOrVectorOf<type>:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "$lhs `,` $rhs custom<LLVMOpAttrs>(attr-dict) `:` type($res)";
}
class LLVM_IntArithmeticOp<string mnemonic, string builderFunc,
                           list<Trait> traits = []> :
    LLVM_ArithmeticOpBase<AnyInteger, mnemonic, builderFunc, traits> {
  let arguments = commonArgs;
}
class LLVM_FloatArithmeticOp<string mnemonic, string builderFunc,
                             list<Trait> traits = []> :
    LLVM_ArithmeticOpBase<LLVM_AnyFloat, mnemonic, builderFunc,
    !listconcat([DeclareOpInterfaceMethods<FastmathFlagsInterface>], traits)> {
  dag fmfArg = (ins DefaultValuedAttr<LLVM_FMFAttr, "{}">:$fastmathFlags);
  let arguments = !con(commonArgs, fmfArg);
}

// Class for arithmetic unary operations.
class LLVM_UnaryFloatArithmeticOp<Type type, string mnemonic,
                             string builderFunc, list<Trait> traits = []> :
    LLVM_Op<mnemonic,
           !listconcat([NoSideEffect, SameOperandsAndResultType, DeclareOpInterfaceMethods<FastmathFlagsInterface>], traits)>,
    LLVM_Builder<"$res = builder." # builderFunc # "($operand);"> {
  let arguments = (ins type:$operand, DefaultValuedAttr<LLVM_FMFAttr, "{}">:$fastmathFlags);
  let results = (outs type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "$operand custom<LLVMOpAttrs>(attr-dict) `:` type($res)";
}

// Integer binary operations.
def LLVM_AddOp : LLVM_IntArithmeticOp<"add", "CreateAdd", [Commutative]>;
def LLVM_SubOp : LLVM_IntArithmeticOp<"sub", "CreateSub">;
def LLVM_MulOp : LLVM_IntArithmeticOp<"mul", "CreateMul", [Commutative]>;
def LLVM_UDivOp : LLVM_IntArithmeticOp<"udiv", "CreateUDiv">;
def LLVM_SDivOp : LLVM_IntArithmeticOp<"sdiv", "CreateSDiv">;
def LLVM_URemOp : LLVM_IntArithmeticOp<"urem", "CreateURem">;
def LLVM_SRemOp : LLVM_IntArithmeticOp<"srem", "CreateSRem">;
def LLVM_AndOp : LLVM_IntArithmeticOp<"and", "CreateAnd">;
def LLVM_OrOp : LLVM_IntArithmeticOp<"or", "CreateOr">;
def LLVM_XOrOp : LLVM_IntArithmeticOp<"xor", "CreateXor">;
def LLVM_ShlOp : LLVM_IntArithmeticOp<"shl", "CreateShl">;
def LLVM_LShrOp : LLVM_IntArithmeticOp<"lshr", "CreateLShr">;
def LLVM_AShrOp : LLVM_IntArithmeticOp<"ashr", "CreateAShr">;

// Predicate for integer comparisons.
def ICmpPredicateEQ  : I64EnumAttrCase<"eq", 0>;
def ICmpPredicateNE  : I64EnumAttrCase<"ne", 1>;
def ICmpPredicateSLT : I64EnumAttrCase<"slt", 2>;
def ICmpPredicateSLE : I64EnumAttrCase<"sle", 3>;
def ICmpPredicateSGT : I64EnumAttrCase<"sgt", 4>;
def ICmpPredicateSGE : I64EnumAttrCase<"sge", 5>;
def ICmpPredicateULT : I64EnumAttrCase<"ult", 6>;
def ICmpPredicateULE : I64EnumAttrCase<"ule", 7>;
def ICmpPredicateUGT : I64EnumAttrCase<"ugt", 8>;
def ICmpPredicateUGE : I64EnumAttrCase<"uge", 9>;
def ICmpPredicate : I64EnumAttr<
    "ICmpPredicate",
    "llvm.icmp comparison predicate",
    [ICmpPredicateEQ, ICmpPredicateNE, ICmpPredicateSLT, ICmpPredicateSLE,
     ICmpPredicateSGT, ICmpPredicateSGE, ICmpPredicateULT, ICmpPredicateULE,
     ICmpPredicateUGT, ICmpPredicateUGE]> {
  let cppNamespace = "::mlir::LLVM";
}

// Other integer operations.
def LLVM_ICmpOp : LLVM_Op<"icmp", [NoSideEffect]> {
  let arguments = (ins ICmpPredicate:$predicate,
                   AnyTypeOf<[LLVM_ScalarOrVectorOf<AnyInteger>, LLVM_ScalarOrVectorOf<LLVM_AnyPointer>]>:$lhs,
                   AnyTypeOf<[LLVM_ScalarOrVectorOf<AnyInteger>, LLVM_ScalarOrVectorOf<LLVM_AnyPointer>]>:$rhs);
  let results = (outs LLVM_ScalarOrVectorOf<I1>:$res);
  let llvmBuilder = [{
    $res = builder.CreateICmp(getLLVMCmpPredicate($predicate), $lhs, $rhs);
  }];
  let builders = [
    OpBuilder<(ins "ICmpPredicate":$predicate, "Value":$lhs, "Value":$rhs),
    [{
      build($_builder, $_state, IntegerType::get(lhs.getType().getContext(), 1),
            predicate, lhs, rhs);
    }]>];
  let hasCustomAssemblyFormat = 1;
}

// Predicate for float comparisons
def FCmpPredicateFALSE  : I64EnumAttrCase<"_false", 0>;
def FCmpPredicateOEQ    : I64EnumAttrCase<"oeq", 1>;
def FCmpPredicateOGT    : I64EnumAttrCase<"ogt", 2>;
def FCmpPredicateOGE    : I64EnumAttrCase<"oge", 3>;
def FCmpPredicateOLT    : I64EnumAttrCase<"olt", 4>;
def FCmpPredicateOLE    : I64EnumAttrCase<"ole", 5>;
def FCmpPredicateONE    : I64EnumAttrCase<"one", 6>;
def FCmpPredicateORD    : I64EnumAttrCase<"ord", 7>;
def FCmpPredicateUEQ    : I64EnumAttrCase<"ueq", 8>;
def FCmpPredicateUGT    : I64EnumAttrCase<"ugt", 9>;
def FCmpPredicateUGE    : I64EnumAttrCase<"uge", 10>;
def FCmpPredicateULT    : I64EnumAttrCase<"ult", 11>;
def FCmpPredicateULE    : I64EnumAttrCase<"ule", 12>;
def FCmpPredicateUNE    : I64EnumAttrCase<"une", 13>;
def FCmpPredicateUNO    : I64EnumAttrCase<"uno", 14>;
def FCmpPredicateTRUE   : I64EnumAttrCase<"_true", 15>;

def FCmpPredicate : I64EnumAttr<
    "FCmpPredicate",
    "llvm.fcmp comparison predicate",
    [FCmpPredicateFALSE, FCmpPredicateOEQ, FCmpPredicateOGT, FCmpPredicateOGE,
     FCmpPredicateOLT, FCmpPredicateOLE, FCmpPredicateONE, FCmpPredicateORD,
     FCmpPredicateUEQ, FCmpPredicateUGT, FCmpPredicateUGE, FCmpPredicateULT,
     FCmpPredicateULE, FCmpPredicateUNE, FCmpPredicateUNO, FCmpPredicateTRUE
    ]> {
  let cppNamespace = "::mlir::LLVM";
}

// Other floating-point operations.
def LLVM_FCmpOp : LLVM_Op<"fcmp", [
    NoSideEffect, DeclareOpInterfaceMethods<FastmathFlagsInterface>]> {
  let arguments = (ins FCmpPredicate:$predicate,
                   LLVM_ScalarOrVectorOf<LLVM_AnyFloat>:$lhs,
                   LLVM_ScalarOrVectorOf<LLVM_AnyFloat>:$rhs,
                   DefaultValuedAttr<LLVM_FMFAttr, "{}">:$fastmathFlags);
  let results = (outs LLVM_ScalarOrVectorOf<I1>:$res);
  let llvmBuilder = [{
    $res = builder.CreateFCmp(getLLVMCmpPredicate($predicate), $lhs, $rhs);
  }];
  let hasCustomAssemblyFormat = 1;
}

// Floating point binary operations.
def LLVM_FAddOp : LLVM_FloatArithmeticOp<"fadd", "CreateFAdd">;
def LLVM_FSubOp : LLVM_FloatArithmeticOp<"fsub", "CreateFSub">;
def LLVM_FMulOp : LLVM_FloatArithmeticOp<"fmul", "CreateFMul">;
def LLVM_FDivOp : LLVM_FloatArithmeticOp<"fdiv", "CreateFDiv">;
def LLVM_FRemOp : LLVM_FloatArithmeticOp<"frem", "CreateFRem">;
def LLVM_FNegOp : LLVM_UnaryFloatArithmeticOp<
  LLVM_ScalarOrVectorOf<LLVM_AnyFloat>, "fneg", "CreateFNeg">;

// Common code definition that is used to verify and set the alignment attribute
// of LLVM ops that accept such an attribute.
class MemoryOpWithAlignmentBase {
  code setAlignmentCode = [{
    if ($alignment.hasValue()) {
      auto align = $alignment.getValue();
      if (align != 0)
        inst->setAlignment(llvm::Align(align));
    }
  }];
}

// Code definition that is used for nontemporal metadata creation.
class MemoryOpWithAlignmentAndAttributes : MemoryOpWithAlignmentBase {
  code setNonTemporalMetadataCode = [{
    if ($nontemporal) {
      llvm::Module *module = builder.GetInsertBlock()->getModule();
      llvm::MDNode *metadata = llvm::MDNode::get(
          inst->getContext(), llvm::ConstantAsMetadata::get(
              builder.getInt32(1)));
      inst->setMetadata(module->getMDKindID("nontemporal"), metadata);
    }
  }];

  code setAccessGroupsMetadataCode = [{
    moduleTranslation.setAccessGroupsMetadata(op, inst);
  }];

  code setAliasScopeMetadataCode = [{
    moduleTranslation.setAliasScopeMetadata(op, inst);
  }];
}

// Memory-related operations.
def LLVM_AllocaOp : LLVM_Op<"alloca">, MemoryOpWithAlignmentBase {
  let arguments = (ins AnyInteger:$arraySize,
                   OptionalAttr<I64Attr>:$alignment);
  let results = (outs Res<LLVM_AnyPointer, "", [MemAlloc<AutomaticAllocationScopeResource>]>:$res);
  string llvmBuilder = [{
    auto addrSpace = $_resultType->getPointerAddressSpace();
    auto *inst = builder.CreateAlloca(
        $_resultType->getPointerElementType(), addrSpace, $arraySize);
    }] # setAlignmentCode # [{
    $res = inst;
  }];
  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$arraySize,
      "unsigned":$alignment),
    [{
      if (alignment == 0)
        return build($_builder, $_state, resultType, arraySize, IntegerAttr());
      build($_builder, $_state, resultType, arraySize,
        $_builder.getI64IntegerAttr(alignment));
  }]>];
  let hasCustomAssemblyFormat = 1;
}

def LLVM_GEPOp : LLVM_Op<"getelementptr", [NoSideEffect]> {
  let arguments = (ins LLVM_ScalarOrVectorOf<LLVM_AnyPointer>:$base,
                   Variadic<LLVM_ScalarOrVectorOf<AnyInteger>>:$indices,
                   I32ElementsAttr:$structIndices);
  let results = (outs LLVM_ScalarOrVectorOf<LLVM_AnyPointer>:$res);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$basePtr, "ValueRange":$indices,
               CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "Type":$resultType, "Value":$basePtr, "ValueRange":$indices,
               "ArrayRef<int32_t>":$structIndices,
               CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];
  let llvmBuilder = [{
    SmallVector<llvm::Value *> indices;
    indices.reserve($structIndices.size());
    unsigned operandIdx = 0;
    for (int32_t structIndex : $structIndices.getValues<int32_t>()) {
      if (structIndex == GEPOp::kDynamicIndex)
        indices.push_back($indices[operandIdx++]);
      else
        indices.push_back(builder.getInt32(structIndex));
    }
    $res = builder.CreateGEP(
      $base->getType()->getPointerElementType(), $base, indices);
  }];
  let assemblyFormat = [{
    $base `[` custom<GEPIndices>($indices, $structIndices) `]` attr-dict
    `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    constexpr static int kDynamicIndex = std::numeric_limits<int32_t>::min();
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

def LLVM_LoadOp : LLVM_Op<"load">, MemoryOpWithAlignmentAndAttributes {
  let arguments = (ins Arg<LLVM_PointerTo<LLVM_LoadableType>, "", [MemRead]>:$addr,
                   OptionalAttr<SymbolRefArrayAttr>:$access_groups,
                   OptionalAttr<SymbolRefArrayAttr>:$alias_scopes,
                   OptionalAttr<SymbolRefArrayAttr>:$noalias_scopes,
                   OptionalAttr<I64Attr>:$alignment, UnitAttr:$volatile_,
                   UnitAttr:$nontemporal);
  let results = (outs LLVM_LoadableType:$res);
  string llvmBuilder = [{
    auto *inst = builder.CreateLoad(
      $addr->getType()->getPointerElementType(), $addr, $volatile_);
  }] # setAlignmentCode
     # setNonTemporalMetadataCode
     # setAccessGroupsMetadataCode
     # setAliasScopeMetadataCode
     # [{
    $res = inst;
  }];
  let builders = [
    OpBuilder<(ins "Value":$addr, CArg<"unsigned", "0">:$alignment,
      CArg<"bool", "false">:$isVolatile, CArg<"bool", "false">:$isNonTemporal),
    [{
      auto type = addr.getType().cast<LLVMPointerType>().getElementType();
      build($_builder, $_state, type, addr, alignment, isVolatile, isNonTemporal);
    }]>,
    OpBuilder<(ins "Type":$t, "Value":$addr,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isVolatile,
      CArg<"bool", "false">:$isNonTemporal)>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LLVM_StoreOp : LLVM_Op<"store">, MemoryOpWithAlignmentAndAttributes {
  let arguments = (ins LLVM_LoadableType:$value,
                   Arg<LLVM_PointerTo<LLVM_LoadableType>,"",[MemWrite]>:$addr,
                   OptionalAttr<SymbolRefArrayAttr>:$access_groups,
                   OptionalAttr<SymbolRefArrayAttr>:$alias_scopes,
                   OptionalAttr<SymbolRefArrayAttr>:$noalias_scopes,
                   OptionalAttr<I64Attr>:$alignment, UnitAttr:$volatile_,
                   UnitAttr:$nontemporal);
  string llvmBuilder = [{
    auto *inst = builder.CreateStore($value, $addr, $volatile_);
  }] # setAlignmentCode
     # setNonTemporalMetadataCode
     # setAccessGroupsMetadataCode
     # setAliasScopeMetadataCode;
  let builders = [
    OpBuilder<(ins "Value":$value, "Value":$addr,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isVolatile,
      CArg<"bool", "false">:$isNonTemporal)>
    ];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// Casts.
class LLVM_CastOp<string mnemonic, string builderFunc, Type type,
                  Type resultType, list<Trait> traits = []> :
    LLVM_Op<mnemonic, !listconcat([NoSideEffect], traits)>,
    LLVM_Builder<"$res = builder." # builderFunc # "($arg, $_resultType);"> {
  let arguments = (ins type:$arg);
  let results = (outs resultType:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
}
def LLVM_BitcastOp : LLVM_CastOp<"bitcast", "CreateBitCast",
                                 LLVM_AnyNonAggregate, LLVM_AnyNonAggregate> {
  let hasFolder = 1;
}
def LLVM_AddrSpaceCastOp : LLVM_CastOp<"addrspacecast", "CreateAddrSpaceCast",
                                       LLVM_ScalarOrVectorOf<LLVM_AnyPointer>,
                                       LLVM_ScalarOrVectorOf<LLVM_AnyPointer>> {
  let hasFolder = 1;
}
def LLVM_IntToPtrOp : LLVM_CastOp<"inttoptr", "CreateIntToPtr",
                                  LLVM_ScalarOrVectorOf<AnyInteger>,
                                  LLVM_ScalarOrVectorOf<LLVM_AnyPointer>>;
def LLVM_PtrToIntOp : LLVM_CastOp<"ptrtoint", "CreatePtrToInt",
                                  LLVM_ScalarOrVectorOf<LLVM_AnyPointer>,
                                  LLVM_ScalarOrVectorOf<AnyInteger>>;
def LLVM_SExtOp : LLVM_CastOp<"sext", "CreateSExt",
                              LLVM_ScalarOrVectorOf<AnyInteger>,
                              LLVM_ScalarOrVectorOf<AnyInteger>>;
def LLVM_ZExtOp : LLVM_CastOp<"zext", "CreateZExt",
                              LLVM_ScalarOrVectorOf<AnyInteger>,
                              LLVM_ScalarOrVectorOf<AnyInteger>>;
def LLVM_TruncOp : LLVM_CastOp<"trunc", "CreateTrunc",
                               LLVM_ScalarOrVectorOf<AnyInteger>,
                               LLVM_ScalarOrVectorOf<AnyInteger>>;
def LLVM_SIToFPOp : LLVM_CastOp<"sitofp", "CreateSIToFP",
                                LLVM_ScalarOrVectorOf<AnyInteger>,
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>>;
def LLVM_UIToFPOp : LLVM_CastOp<"uitofp", "CreateUIToFP",
                                LLVM_ScalarOrVectorOf<AnyInteger>,
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>>;
def LLVM_FPToSIOp : LLVM_CastOp<"fptosi", "CreateFPToSI",
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>,
                                LLVM_ScalarOrVectorOf<AnyInteger>>;
def LLVM_FPToUIOp : LLVM_CastOp<"fptoui", "CreateFPToUI",
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>,
                                LLVM_ScalarOrVectorOf<AnyInteger>>;
def LLVM_FPExtOp : LLVM_CastOp<"fpext", "CreateFPExt",
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>,
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>>;
def LLVM_FPTruncOp : LLVM_CastOp<"fptrunc", "CreateFPTrunc",
                                 LLVM_ScalarOrVectorOf<LLVM_AnyFloat>,
                                 LLVM_ScalarOrVectorOf<LLVM_AnyFloat>>;

// Call-related operations.
def LLVM_InvokeOp : LLVM_Op<"invoke", [
                      AttrSizedOperandSegments,
                      DeclareOpInterfaceMethods<BranchOpInterface>,
                      Terminator]> {
  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$callee,
                   Variadic<LLVM_Type>:$callee_operands,
                   Variadic<LLVM_Type>:$normalDestOperands,
                   Variadic<LLVM_Type>:$unwindDestOperands);
  let results = (outs Variadic<LLVM_Type>);
  let successors = (successor AnySuccessor:$normalDest,
                              AnySuccessor:$unwindDest);

  let builders = [
    OpBuilder<(ins "TypeRange":$tys, "FlatSymbolRefAttr":$callee,
      "ValueRange":$ops, "Block*":$normal, "ValueRange":$normalOps,
      "Block*":$unwind, "ValueRange":$unwindOps),
    [{
      $_state.addAttribute("callee", callee);
      build($_builder, $_state, tys, ops, normal, normalOps, unwind, unwindOps);
    }]>,
    OpBuilder<(ins "TypeRange":$tys, "ValueRange":$ops, "Block*":$normal,
      "ValueRange":$normalOps, "Block*":$unwind, "ValueRange":$unwindOps),
    [{
      build($_builder, $_state, tys, /*callee=*/FlatSymbolRefAttr(), ops, normalOps,
            unwindOps, normal, unwind);
    }]>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LLVM_LandingpadOp : LLVM_Op<"landingpad"> {
  let arguments = (ins UnitAttr:$cleanup, Variadic<LLVM_Type>);
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LLVM_CallOp : LLVM_Op<"call",
                          [DeclareOpInterfaceMethods<FastmathFlagsInterface>]> {
  let summary = "Call to an LLVM function.";
  let description = [{


    In LLVM IR, functions may return either 0 or 1 value. LLVM IR dialect
    implements this behavior by providing a variadic `call` operation for 0- and
    1-result functions. Even though MLIR supports multi-result functions, LLVM
    IR dialect disallows them.

    The `call` instruction supports both direct and indirect calls. Direct calls
    start with a function name (`@`-prefixed) and indirect calls start with an
    SSA value (`%`-prefixed). The direct callee, if present, is stored as a
    function attribute `callee`. The trailing type of the instruction is always
    the MLIR function type, which may be different from the indirect callee that
    has the wrapped LLVM IR function type.

    Examples:

    ```mlir
    // Direct call without arguments and with one result.
    %0 = llvm.call @foo() : () -> (f32)

    // Direct call with arguments and without a result.
    llvm.call @bar(%0) : (f32) -> ()

    // Indirect call with an argument and without a result.
    llvm.call %1(%0) : (f32) -> ()
    ```
  }];
  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$callee,
                   Variadic<LLVM_Type>,
                   DefaultValuedAttr<LLVM_FMFAttr, "{}">:$fastmathFlags);
  let results = (outs Variadic<LLVM_Type>);
  let builders = [
    OpBuilder<(ins "LLVMFuncOp":$func, "ValueRange":$operands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes), [{
      Type resultType = func.getType().getReturnType();
      if (!resultType.isa<LLVM::LLVMVoidType>())
        $_state.addTypes(resultType);
      $_state.addAttribute("callee", SymbolRefAttr::get(func));
      $_state.addAttributes(attributes);
      $_state.addOperands(operands);
    }]>,
    OpBuilder<(ins "TypeRange":$results, "StringAttr":$callee,
                   CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, results, SymbolRefAttr::get(callee), operands);
    }]>,
    OpBuilder<(ins "TypeRange":$results, "StringRef":$callee,
                   CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, results,
            StringAttr::get($_builder.getContext(), callee), operands);
    }]>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
def LLVM_ExtractElementOp : LLVM_Op<"extractelement", [NoSideEffect]> {
  let arguments = (ins LLVM_AnyVector:$vector, AnyInteger:$position);
  let results = (outs LLVM_Type:$res);
  string llvmBuilder = [{
    $res = builder.CreateExtractElement($vector, $position);
  }];
  let builders = [
    OpBuilder<(ins "Value":$vector, "Value":$position,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
def LLVM_ExtractValueOp : LLVM_Op<"extractvalue", [NoSideEffect]> {
  let arguments = (ins LLVM_AnyAggregate:$container, ArrayAttr:$position);
  let results = (outs LLVM_Type:$res);
  string llvmBuilder = [{
    $res = builder.CreateExtractValue($container, extractPosition($position));
  }];
  let builders = [LLVM_OneResultOpBuilder];
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}
def LLVM_InsertElementOp : LLVM_Op<"insertelement", [NoSideEffect]> {
  let arguments = (ins LLVM_AnyVector:$vector, LLVM_PrimitiveType:$value,
                   AnyInteger:$position);
  let results = (outs LLVM_AnyVector:$res);
  string llvmBuilder = [{
    $res = builder.CreateInsertElement($vector, $value, $position);
  }];
  let builders = [LLVM_OneResultOpBuilder];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
def LLVM_InsertValueOp : LLVM_Op<"insertvalue", [NoSideEffect]> {
  let arguments = (ins LLVM_AnyAggregate:$container, LLVM_PrimitiveType:$value,
                   ArrayAttr:$position);
  let results = (outs LLVM_AnyAggregate:$res);
  string llvmBuilder = [{
    $res = builder.CreateInsertValue($container, $value,
                                     extractPosition($position));
  }];
  let builders = [
    OpBuilder<(ins "Value":$container, "Value":$value, "ArrayAttr":$position),
    [{
      build($_builder, $_state, container.getType(), container, value, position);
    }]>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
def LLVM_ShuffleVectorOp : LLVM_Op<"shufflevector", [NoSideEffect]> {
  let arguments = (ins LLVM_AnyVector:$v1, LLVM_AnyVector:$v2, ArrayAttr:$mask);
  let results = (outs LLVM_AnyVector:$res);
  string llvmBuilder = [{
      SmallVector<unsigned, 4> position = extractPosition($mask);
      SmallVector<int, 4> mask(position.begin(), position.end());
      $res = builder.CreateShuffleVector($v1, $v2, mask);
  }];
  let builders = [
    OpBuilder<(ins "Value":$v1, "Value":$v2, "ArrayAttr":$mask,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// Misc operations.
def LLVM_SelectOp
    : LLVM_Op<"select",
          [NoSideEffect, AllTypesMatch<["trueValue", "falseValue", "res"]>]>,
      LLVM_Builder<
          "$res = builder.CreateSelect($condition, $trueValue, $falseValue);"> {
  let arguments = (ins LLVM_ScalarOrVectorOf<I1>:$condition,
                   LLVM_Type:$trueValue, LLVM_Type:$falseValue);
  let results = (outs LLVM_Type:$res);
  let assemblyFormat = "operands attr-dict `:` type($condition) `,` type($res)";
}
def LLVM_FreezeOp : LLVM_Op<"freeze", [SameOperandsAndResultType]> {
  let arguments = (ins LLVM_Type:$val);
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "$val attr-dict `:` type($val)";
  string llvmBuilder = "builder.CreateFreeze($val);";
}

// Terminators.
def LLVM_BrOp : LLVM_TerminatorOp<"br",
    [DeclareOpInterfaceMethods<BranchOpInterface>, NoSideEffect]> {
  let arguments = (ins Variadic<LLVM_Type>:$destOperands);
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
  let builders = [LLVM_TerminatorPassthroughOpBuilder];
}
def LLVM_CondBrOp : LLVM_TerminatorOp<"cond_br",
    [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BranchOpInterface>,
     NoSideEffect]> {
  let arguments = (ins I1:$condition,
                   Variadic<LLVM_Type>:$trueDestOperands,
                   Variadic<LLVM_Type>:$falseDestOperands,
                   OptionalAttr<ElementsAttr>:$branch_weights);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
  let assemblyFormat = [{
    $condition ( `weights` `(` $branch_weights^ `)` )? `,`
    $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
    $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Value":$condition, "Block *":$trueDest,
      "ValueRange":$trueOperands, "Block *":$falseDest,
      "ValueRange":$falseOperands,
      CArg<"Optional<std::pair<uint32_t, uint32_t>>", "{}">:$weights),
    [{
        ElementsAttr weightsAttr;
        if (weights) {
          weightsAttr =
              $_builder.getI32VectorAttr({static_cast<int32_t>(weights->first),
                                       static_cast<int32_t>(weights->second)});
        }
        build($_builder, $_state, condition, trueOperands, falseOperands, weightsAttr,
              trueDest, falseDest);
  }]>,
  OpBuilder<(ins "Value":$condition, "Block *":$trueDest,
    "Block *":$falseDest, CArg<"ValueRange", "{}">:$falseOperands),
  [{
      build($_builder, $_state, condition, trueDest, ValueRange(), falseDest,
            falseOperands);
  }]>, LLVM_TerminatorPassthroughOpBuilder];
}
def LLVM_ReturnOp : LLVM_TerminatorOp<"return", [NoSideEffect]> {
  let arguments = (ins Variadic<LLVM_Type>:$args);
  string llvmBuilder = [{
    if ($_numOperands != 0)
      builder.CreateRet($args[0]);
    else
      builder.CreateRetVoid();
  }];

  let assemblyFormat = "attr-dict ($args^ `:` type($args))?";
  let hasVerifier = 1;
}
def LLVM_ResumeOp : LLVM_TerminatorOp<"resume", []> {
  let arguments = (ins LLVM_Type:$value);
  string llvmBuilder = [{ builder.CreateResume($value); }];
  let assemblyFormat = "$value attr-dict `:` type($value)";
  let hasVerifier = 1;
}
def LLVM_UnreachableOp : LLVM_TerminatorOp<"unreachable", []> {
  string llvmBuilder = [{ builder.CreateUnreachable(); }];
  let assemblyFormat = "attr-dict";
}

def LLVM_SwitchOp : LLVM_TerminatorOp<"switch",
    [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BranchOpInterface>,
     NoSideEffect]> {
  let arguments = (ins
    AnyInteger:$value,
    Variadic<AnyType>:$defaultOperands,
    VariadicOfVariadic<AnyType, "case_operand_segments">:$caseOperands,
    OptionalAttr<ElementsAttr>:$case_values,
    ElementsAttr:$case_operand_segments,
    OptionalAttr<ElementsAttr>:$branch_weights
  );
  let successors = (successor
    AnySuccessor:$defaultDestination,
    VariadicSuccessor<AnySuccessor>:$caseDestinations
  );

  let assemblyFormat = [{
    $value `:` type($value) `,`
    $defaultDestination (`(` $defaultOperands^ `:` type($defaultOperands) `)`)?
    `[` `\n` custom<SwitchOpCases>(ref(type($value)), $case_values, $caseDestinations,
                                   $caseOperands, type($caseOperands)) `]`
    attr-dict
  }];
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$value,
      "Block *":$defaultDestination,
      "ValueRange":$defaultOperands,
      CArg<"ArrayRef<int32_t>", "{}">:$caseValues,
      CArg<"BlockRange", "{}">:$caseDestinations,
      CArg<"ArrayRef<ValueRange>", "{}">:$caseOperands,
      CArg<"ArrayRef<int32_t>", "{}">:$branchWeights)>,
    LLVM_TerminatorPassthroughOpBuilder
  ];

  let extraClassDeclaration = [{
    /// Return the operands for the case destination block at the given index.
    OperandRange getCaseOperands(unsigned index) {
      return getCaseOperands()[index];
    }

    /// Return a mutable range of operands for the case destination block at the
    /// given index.
    MutableOperandRange getCaseOperandsMutable(unsigned index) {
      return getCaseOperandsMutable()[index];
    }
  }];
}

////////////////////////////////////////////////////////////////////////////////
// Auxiliary operations (do not appear in LLVM IR but necessary for the dialect
// to work correctly).
////////////////////////////////////////////////////////////////////////////////

// Linkage attribute is used on functions and globals. The order follows that of
// https://llvm.org/docs/LangRef.html#linkage-types. The names are equivalent to
// visible names in the IR rather than to enum values names in llvm::GlobalValue
// since the latter is easier to change.
def LinkagePrivate
    : LLVM_EnumAttrCase<"Private", "private", "PrivateLinkage", 0>;
def LinkageInternal
    : LLVM_EnumAttrCase<"Internal", "internal", "InternalLinkage", 1>;
def LinkageAvailableExternally
    : LLVM_EnumAttrCase<"AvailableExternally", "available_externally",
                        "AvailableExternallyLinkage", 2>;
def LinkageLinkonce
    : LLVM_EnumAttrCase<"Linkonce", "linkonce", "LinkOnceAnyLinkage", 3>;
def LinkageWeak
    : LLVM_EnumAttrCase<"Weak", "weak", "WeakAnyLinkage", 4>;
def LinkageCommon
    : LLVM_EnumAttrCase<"Common", "common", "CommonLinkage", 5>;
def LinkageAppending
    : LLVM_EnumAttrCase<"Appending", "appending", "AppendingLinkage", 6>;
def LinkageExternWeak
   : LLVM_EnumAttrCase<"ExternWeak", "extern_weak", "ExternalWeakLinkage", 7>;
def LinkageLinkonceODR
    : LLVM_EnumAttrCase<"LinkonceODR", "linkonce_odr", "LinkOnceODRLinkage", 8>;
def LinkageWeakODR
    : LLVM_EnumAttrCase<"WeakODR", "weak_odr", "WeakODRLinkage", 9>;
def LinkageExternal
    : LLVM_EnumAttrCase<"External", "external", "ExternalLinkage", 10>;

def LinkageEnum : LLVM_EnumAttr<
    "Linkage",
    "::llvm::GlobalValue::LinkageTypes",
    "LLVM linkage types",
    [LinkagePrivate, LinkageInternal, LinkageAvailableExternally,
     LinkageLinkonce, LinkageWeak, LinkageCommon, LinkageAppending,
     LinkageExternWeak, LinkageLinkonceODR, LinkageWeakODR, LinkageExternal]> {
  let cppNamespace = "::mlir::LLVM::linkage";
}

def Linkage : DialectAttr<
    LLVM_Dialect,
    CPred<"$_self.isa<::mlir::LLVM::LinkageAttr>()">,
    "LLVM Linkage specification"> {
  let storageType = "::mlir::LLVM::LinkageAttr";
  let returnType = "::mlir::LLVM::Linkage";
  let convertFromStorage = "$_self.getLinkage()";
  let constBuilderCall =
          "::mlir::LLVM::LinkageAttr::get($_builder.getContext(), $0)";
}


def UnnamedAddrNone : LLVM_EnumAttrCase<"None", "", "None", 0>;
def UnnamedAddrLocal : LLVM_EnumAttrCase<"Local", "local_unnamed_addr", "Local", 1>;
def UnnamedAddrGlobal : LLVM_EnumAttrCase<"Global", "unnamed_addr", "Global", 2>;

def UnnamedAddr : LLVM_EnumAttr<
    "UnnamedAddr",
    "::llvm::GlobalValue::UnnamedAddr",
    "LLVM GlobalValue UnnamedAddr",
    [UnnamedAddrNone, UnnamedAddrLocal, UnnamedAddrGlobal]> {
  let cppNamespace = "::mlir::LLVM";
}

def LLVM_AddressOfOp : LLVM_Op<"mlir.addressof", [NoSideEffect]> {
  let arguments = (ins FlatSymbolRefAttr:$global_name);
  let results = (outs LLVM_Type:$res);

  let summary = "Creates a pointer pointing to a global or a function";

  let description = [{
    Creates an SSA value containing a pointer to a global variable or constant
    defined by `llvm.mlir.global`. The global value can be defined after its
    first referenced. If the global value is a constant, storing into it is not
    allowed.

    Examples:

    ```mlir
    func @foo() {
      // Get the address of a global variable.
      %0 = llvm.mlir.addressof @const : !llvm.ptr<i32>

      // Use it as a regular pointer.
      %1 = llvm.load %0 : !llvm.ptr<i32>

      // Get the address of a function.
      %2 = llvm.mlir.addressof @foo : !llvm.ptr<func<void ()>>

      // The function address can be used for indirect calls.
      llvm.call %2() : () -> ()
    }

    // Define the global.
    llvm.mlir.global @const(42 : i32) : i32
    ```
  }];

  let builders = [
    OpBuilder<(ins "GlobalOp":$global,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      build($_builder, $_state,
            LLVM::LLVMPointerType::get(global.getType(), global.getAddrSpace()),
            global.getSymName());
      $_state.addAttributes(attrs);
    }]>,
    OpBuilder<(ins "LLVMFuncOp":$func,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      build($_builder, $_state,
            LLVM::LLVMPointerType::get(func.getType()), func.getName());
      $_state.addAttributes(attrs);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the llvm.mlir.global operation that defined the value referenced
    /// here.
    GlobalOp getGlobal();

    /// Return the llvm.func operation that is referenced here.
    LLVMFuncOp getFunction();
  }];

  let assemblyFormat = "$global_name attr-dict `:` type($res)";
  let hasVerifier = 1;
}

def LLVM_MetadataOp : LLVM_Op<"metadata", [
   NoRegionArguments, SymbolTable, Symbol
]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name
  );
  let summary = "LLVM dialect metadata.";
  let description = [{
    llvm.metadata op defines one or more metadata nodes.

    Example:
      llvm.metadata @metadata {
        llvm.access_group @group1
        llvm.access_group @group2
        llvm.return
      }
  }];
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def LLVM_AliasScopeDomainMetadataOp : LLVM_Op<"alias_scope_domain", [
  HasParent<"MetadataOp">, Symbol
]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$description
  );
  let summary = "LLVM dialect alias.scope domain metadata.";
  let description = [{
    Defines a domain that may be associated with an alias scope.

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#noalias-and-alias-scope-metadata
  }];
  let assemblyFormat = "$sym_name attr-dict";
}

def LLVM_AliasScopeMetadataOp : LLVM_Op<"alias_scope", [
  HasParent<"MetadataOp">, Symbol
]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$domain,
    OptionalAttr<StrAttr>:$description
  );
  let summary = "LLVM dialect alias.scope metadata.";
  let description = [{
    Defines an alias scope that can be attached to a memory-accessing operation.
    Such scopes can be used in combination with `noalias` metadata to indicate
    that sets of memory-affecting operations in one scope do not alias with
    memory-affecting operations in another scope.

    Example:
      module {
        llvm.func @foo(%ptr1 : !llvm.ptr<i32>) {
            %c0 = llvm.mlir.constant(0 : i32) : i32
            %c4 = llvm.mlir.constant(4 : i32) : i32
            %1 = llvm.ptrtoint %ptr1 : !llvm.ptr<i32> to i32
            %2 = llvm.add %1, %c1 : i32
            %ptr2 = llvm.inttoptr %2 : i32 to !llvm.ptr<i32>
            llvm.store %c0, %ptr1 { alias_scopes = [@metadata::@scope1], llvm.noalias = [@metadata::@scope2] } : !llvm.ptr<i32>
            llvm.store %c4, %ptr2 { alias_scopes = [@metadata::@scope2], llvm.noalias = [@metadata::@scope1] } : !llvm.ptr<i32>
            llvm.return
        }

        llvm.metadata @metadata {
          llvm.alias_scope_domain @unused_domain
          llvm.alias_scope_domain @domain { description = "Optional domain description"}
          llvm.alias_scope @scope1 { domain = @domain }
          llvm.alias_scope @scope2 { domain = @domain, description = "Optional scope description" }
          llvm.return
        }
      }

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#noalias-and-alias-scope-metadata
  }];
  let assemblyFormat = "$sym_name attr-dict";
}

def LLVM_AccessGroupMetadataOp : LLVM_Op<"access_group", [
  HasParent<"MetadataOp">, Symbol
]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name
  );
  let summary = "LLVM dialect access group metadata.";
  let description = [{
    Defines an access group metadata that can be attached to any instruction
    that potentially accesses memory. The access group may be attached to a
    memory accessing instruction via the `llvm.access.group` metadata and
    a branch instruction in the loop latch block via the
    `llvm.loop.parallel_accesses` metadata.

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#llvm-access-group-metadata
  }];
  let assemblyFormat = "$sym_name attr-dict";
}

def LLVM_GlobalOp : LLVM_Op<"mlir.global",
    [IsolatedFromAbove, SingleBlockImplicitTerminator<"ReturnOp">, Symbol]> {
  let arguments = (ins
    TypeAttr:$global_type,
    UnitAttr:$constant,
    StrAttr:$sym_name,
    Linkage:$linkage,
    UnitAttr:$dso_local,
    OptionalAttr<AnyAttr>:$value,
    OptionalAttr<I64Attr>:$alignment,
    DefaultValuedAttr<Confined<I32Attr, [IntNonNegative]>, "0">:$addr_space,
    OptionalAttr<UnnamedAddr>:$unnamed_addr,
    OptionalAttr<StrAttr>:$section
  );
  let summary = "LLVM dialect global.";
  let description = [{
    Since MLIR allows for arbitrary operations to be present at the top level,
    global variables are defined using the `llvm.mlir.global` operation. Both
    global constants and variables can be defined, and the value may also be
    initialized in both cases.

    There are two forms of initialization syntax. Simple constants that can be
    represented as MLIR attributes can be given in-line:

    ```mlir
    llvm.mlir.global @variable(32.0 : f32) : f32
    ```

    This initialization and type syntax is similar to `llvm.mlir.constant` and
    may use two types: one for MLIR attribute and another for the LLVM value.
    These types must be compatible.

    More complex constants that cannot be represented as MLIR attributes can be
    given in an initializer region:

    ```mlir
    // This global is initialized with the equivalent of:
    //   i32* getelementptr (i32* @g2, i32 2)
    llvm.mlir.global constant @int_gep() : !llvm.ptr<i32> {
      %0 = llvm.mlir.addressof @g2 : !llvm.ptr<i32>
      %1 = llvm.mlir.constant(2 : i32) : i32
      %2 = llvm.getelementptr %0[%1]
         : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
      // The initializer region must end with `llvm.return`.
      llvm.return %2 : !llvm.ptr<i32>
    }
    ```

    Only one of the initializer attribute or initializer region may be provided.

    `llvm.mlir.global` must appear at top-level of the enclosing module. It uses
    an @-identifier for its value, which will be uniqued by the module with
    respect to other @-identifiers in it.

    Examples:

    ```mlir
    // Global values use @-identifiers.
    llvm.mlir.global constant @cst(42 : i32) : i32

    // Non-constant values must also be initialized.
    llvm.mlir.global @variable(32.0 : f32) : f32

    // Strings are expected to be of wrapped LLVM i8 array type and do not
    // automatically include the trailing zero.
    llvm.mlir.global @string("abc") : !llvm.array<3 x i8>

    // For strings globals, the trailing type may be omitted.
    llvm.mlir.global constant @no_trailing_type("foo bar")

    // A complex initializer is constructed with an initializer region.
    llvm.mlir.global constant @int_gep() : !llvm.ptr<i32> {
      %0 = llvm.mlir.addressof @g2 : !llvm.ptr<i32>
      %1 = llvm.mlir.constant(2 : i32) : i32
      %2 = llvm.getelementptr %0[%1]
         : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
      llvm.return %2 : !llvm.ptr<i32>
    }
    ```

    Similarly to functions, globals have a linkage attribute. In the custom
    syntax, this attribute is placed between `llvm.mlir.global` and the optional
    `constant` keyword. If the attribute is omitted, `external` linkage is
    assumed by default.

    Examples:

    ```mlir
    // A constant with internal linkage will not participate in linking.
    llvm.mlir.global internal constant @cst(42 : i32) : i32

    // By default, "external" linkage is assumed and the global participates in
    // symbol resolution at link-time.
    llvm.mlir.global @glob(0 : f32) : f32

    // Alignment is optional
    llvm.mlir.global private constant @y(dense<1.0> : tensor<8xf32>) : !llvm.array<8 x f32>
    ```

    Like global variables in LLVM IR, globals can have an (optional)
    alignment attribute using keyword `alignment`. The integer value of the
    alignment must be a positive integer that is a power of 2.

    Examples:

    ```mlir
    // Alignment is optional
    llvm.mlir.global private constant @y(dense<1.0> : tensor<8xf32>) { alignment = 32 : i64 } : !llvm.array<8 x f32>
    ```

  }];
  let regions = (region AnyRegion:$initializer);

  let builders = [
    OpBuilder<(ins "Type":$type, "bool":$isConstant, "Linkage":$linkage,
      "StringRef":$name, "Attribute":$value,
      CArg<"uint64_t", "0">:$alignment,
      CArg<"unsigned", "0">:$addrSpace,
      CArg<"bool", "false">:$dsoLocal,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    /// Return the LLVM type of the global.
    Type getType() {
      return getGlobalType();
    }
    /// Return the initializer attribute if it exists, or a null attribute.
    Attribute getValueOrNull() {
      return getValue().getValueOr(Attribute());
    }
    /// Return the initializer region. This may be empty, but if it is not it
    /// terminates in an `llvm.return` op with the initializer value.
    Region &getInitializerRegion() {
      return getOperation()->getRegion(0);
    }
    /// Return the initializer block. If the initializer region is empty this
    /// is nullptr. If it is not nullptr, it terminates with an `llvm.return`
    /// op with the initializer value.
    Block *getInitializerBlock() {
      return getInitializerRegion().empty() ?
        nullptr : &getInitializerRegion().front();
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LLVM_GlobalCtorsOp : LLVM_Op<"mlir.global_ctors", [
                           DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins FlatSymbolRefArrayAttr
                   : $ctors, I32ArrayAttr
                   : $priorities);
  let summary = "LLVM dialect global_ctors.";
  let description = [{
    Specifies a list of constructor functions and priorities. The functions
    referenced by this array will be called in ascending order of priority (i.e.
    lowest first) when the module is loaded. The order of functions with the
    same priority is not defined. This operation is translated to LLVM's
    global_ctors global variable. The initializer functions are run at load
    time. The `data` field present in LLVM's global_ctors variable is not
    modeled here.

    Examples:

    ```mlir
    llvm.mlir.global_ctors {@ctor}

    llvm.func @ctor() {
      ...
      llvm.return
    }
    ```

  }];
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def LLVM_GlobalDtorsOp : LLVM_Op<"mlir.global_dtors", [
                           DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins
    FlatSymbolRefArrayAttr:$dtors,
    I32ArrayAttr:$priorities
  );
  let summary = "LLVM dialect global_dtors.";
  let description = [{
    Specifies a list of destructor functions and priorities. The functions
    referenced by this array will be called in descending order of priority (i.e.
    highest first) when the module is unloaded. The order of functions with the
    same priority is not defined. This operation is translated to LLVM's
    global_dtors global variable. The `data` field present in LLVM's
    global_dtors variable is not modeled here.

    Examples:

    ```mlir
    llvm.func @dtor() {
      llvm.return
    }
    llvm.mlir.global_dtors {@dtor}
    ```

  }];
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def LLVM_LLVMFuncOp : LLVM_Op<"func", [
    AutomaticAllocationScope, IsolatedFromAbove, FunctionOpInterface, Symbol
  ]> {
  let summary = "LLVM dialect function.";

  let description = [{
    MLIR functions are defined by an operation that is not built into the IR
    itself. The LLVM dialect provides an `llvm.func` operation to define
    functions compatible with LLVM IR. These functions have LLVM dialect
    function type but use MLIR syntax to express it. They are required to have
    exactly one result type. LLVM function operation is intended to capture
    additional properties of LLVM functions, such as linkage and calling
    convention, that may be modeled differently by the built-in MLIR function.

    ```mlir
    // The type of @bar is !llvm<"i64 (i64)">
    llvm.func @bar(%arg0: i64) -> i64 {
      llvm.return %arg0 : i64
    }

    // Type type of @foo is !llvm<"void (i64)">
    // !llvm.void type is omitted
    llvm.func @foo(%arg0: i64) {
      llvm.return
    }

    // A function with `internal` linkage.
    llvm.func internal @internal_func() {
      llvm.return
    }
    ```
  }];

  let arguments = (ins DefaultValuedAttr<Linkage, "Linkage::External">:$linkage,
                   UnitAttr:$dso_local,
                   OptionalAttr<FlatSymbolRefAttr>:$personality,
                   OptionalAttr<StrAttr>:$garbageCollector,
                   OptionalAttr<ArrayAttr>:$passthrough);

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "StringRef":$name, "Type":$type,
      CArg<"Linkage", "Linkage::External">:$linkage,
      CArg<"bool", "false">:$dsoLocal,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>
  ];

  let extraClassDeclaration = [{
    // Add an entry block to an empty function, and set up the block arguments
    // to match the signature of the function.
    Block *addEntryBlock();

    LLVMFunctionType getType() {
      return getTypeAttr().getValue().cast<LLVMFunctionType>();
    }
    bool isVarArg() { return getType().isVarArg(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getType().getParams(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getType().getReturnTypes(); }

    /// Hook for FunctionOpInterface, called after verifying that the 'type'
    /// attribute is present.  This can check for preconditions of the
    /// getNumArguments hook not failing.
    LogicalResult verifyType();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LLVM_NullOp
    : LLVM_Op<"mlir.null", [NoSideEffect]>,
      LLVM_Builder<"$res = llvm::ConstantPointerNull::get("
                   "    cast<llvm::PointerType>($_resultType));"> {
  let summary = "Defines a value containing a null pointer to LLVM type.";
  let description = [{
    Unlike LLVM IR, MLIR does not have first-class null pointers. They must be
    explicitly created as SSA values using `llvm.mlir.null`. This operation has
    no operands or attributes, and returns a null value of a wrapped LLVM IR
    pointer type.

    Examples:

    ```mlir
    // Null pointer to i8.
    %0 = llvm.mlir.null : !llvm.ptr<i8>

    // Null pointer to a function with signature void().
    %1 = llvm.mlir.null : !llvm.ptr<func<void ()>>
    ```
  }];

  let results = (outs LLVM_AnyPointer:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "attr-dict `:` type($res)";
}

def LLVM_UndefOp : LLVM_Op<"mlir.undef", [NoSideEffect]>,
                   LLVM_Builder<"$res = llvm::UndefValue::get($_resultType);"> {
  let summary = "Creates an undefined value of LLVM dialect type.";
  let description = [{
    Unlike LLVM IR, MLIR does not have first-class undefined values. Such values
    must be created as SSA values using `llvm.mlir.undef`. This operation has no
    operands or attributes. It creates an undefined value of the specified LLVM
    IR dialect type wrapping an LLVM IR structure type.

    Example:

    ```mlir
    // Create a structure with a 32-bit integer followed by a float.
    %0 = llvm.mlir.undef : !llvm.struct<(i32, f32)>
    ```
  }];
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "attr-dict `:` type($res)";
}

def LLVM_ConstantOp
    : LLVM_Op<"mlir.constant", [NoSideEffect, ConstantLike]>,
      LLVM_Builder<[{$res = getLLVMConstant($_resultType, $value, $_location,
                                            moduleTranslation);}]>
{
  let summary = "Defines a constant of LLVM type.";
  let description = [{
    Unlike LLVM IR, MLIR does not have first-class constant values. Therefore,
    all constants must be created as SSA values before being used in other
    operations. `llvm.mlir.constant` creates such values for scalars and
    vectors. It has a mandatory `value` attribute, which may be an integer,
    floating point attribute; dense or sparse attribute containing integers or
    floats. The type of the attribute is one of the corresponding MLIR builtin
    types. It may be omitted for `i64` and `f64` types that are implied. The
    operation produces a new SSA value of the specified LLVM IR dialect type.
    The type of that value _must_ correspond to the attribute type converted to
    LLVM IR.

    Examples:

    ```mlir
    // Integer constant, internal i32 is mandatory
    %0 = llvm.mlir.constant(42 : i32) : i32

    // It's okay to omit i64.
    %1 = llvm.mlir.constant(42) : i64

    // Floating point constant.
    %2 = llvm.mlir.constant(42.0 : f32) : f32

    // Splat dense vector constant.
    %3 = llvm.mlir.constant(dense<1.0> : vector<4xf32>) : vector<4xf32>
    ```
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "`(` $value `)` attr-dict `:` type($res)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

// Operations that correspond to LLVM intrinsics. With MLIR operation set being
// extendable, there is no reason to introduce a hard boundary between "core"
// operations and intrinsics. However, we systematically prefix them with
// "intr." to avoid potential name clashes.

class LLVM_UnaryIntrinsicOp<string func, list<Trait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)> {
  let arguments = (ins LLVM_Type:$in);
}

class LLVM_BinarySameArgsIntrinsicOp<string func, list<Trait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)> {
  let arguments = (ins LLVM_Type:$a, LLVM_Type:$b);
}

class LLVM_BinaryIntrinsicOp<string func, list<Trait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0,1],
           !listconcat([NoSideEffect], traits)> {
  let arguments = (ins LLVM_Type:$a, LLVM_Type:$b);
}

class LLVM_TernarySameArgsIntrinsicOp<string func, list<Trait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)> {
  let arguments = (ins LLVM_Type:$a, LLVM_Type:$b, LLVM_Type:$c);
}

class LLVM_CountZerosIntrinsicOp<string func, list<Trait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([NoSideEffect], traits)> {
  let arguments = (ins LLVM_Type:$in, I<1>:$zero_undefined);
}

def LLVM_CopySignOp : LLVM_BinarySameArgsIntrinsicOp<"copysign">;
def LLVM_CosOp : LLVM_UnaryIntrinsicOp<"cos">;
def LLVM_ExpOp : LLVM_UnaryIntrinsicOp<"exp">;
def LLVM_Exp2Op : LLVM_UnaryIntrinsicOp<"exp2">;
def LLVM_FAbsOp : LLVM_UnaryIntrinsicOp<"fabs">;
def LLVM_FCeilOp : LLVM_UnaryIntrinsicOp<"ceil">;
def LLVM_FFloorOp : LLVM_UnaryIntrinsicOp<"floor">;
def LLVM_FMAOp : LLVM_TernarySameArgsIntrinsicOp<"fma">;
def LLVM_FMulAddOp : LLVM_TernarySameArgsIntrinsicOp<"fmuladd">;
def LLVM_Log10Op : LLVM_UnaryIntrinsicOp<"log10">;
def LLVM_Log2Op : LLVM_UnaryIntrinsicOp<"log2">;
def LLVM_LogOp : LLVM_UnaryIntrinsicOp<"log">;
def LLVM_Prefetch : LLVM_ZeroResultIntrOp<"prefetch", [0]> {
  let arguments = (ins LLVM_Type:$addr, LLVM_Type:$rw, LLVM_Type:$hint,
                   LLVM_Type:$cache);
}
def LLVM_SinOp : LLVM_UnaryIntrinsicOp<"sin">;
def LLVM_SqrtOp : LLVM_UnaryIntrinsicOp<"sqrt">;
def LLVM_PowOp : LLVM_BinarySameArgsIntrinsicOp<"pow">;
def LLVM_PowIOp : LLVM_BinaryIntrinsicOp<"powi">;
def LLVM_BitReverseOp : LLVM_UnaryIntrinsicOp<"bitreverse">;
def LLVM_CountLeadingZerosOp : LLVM_CountZerosIntrinsicOp<"ctlz">;
def LLVM_CountTrailingZerosOp : LLVM_CountZerosIntrinsicOp<"cttz">;
def LLVM_CtPopOp : LLVM_UnaryIntrinsicOp<"ctpop">;
def LLVM_MaxNumOp : LLVM_BinarySameArgsIntrinsicOp<"maxnum">;
def LLVM_MinNumOp : LLVM_BinarySameArgsIntrinsicOp<"minnum">;
def LLVM_MaximumOp : LLVM_BinarySameArgsIntrinsicOp<"maximum">;
def LLVM_MinimumOp : LLVM_BinarySameArgsIntrinsicOp<"minimum">;
def LLVM_SMaxOp : LLVM_BinarySameArgsIntrinsicOp<"smax">;
def LLVM_SMinOp : LLVM_BinarySameArgsIntrinsicOp<"smin">;
def LLVM_UMaxOp : LLVM_BinarySameArgsIntrinsicOp<"umax">;
def LLVM_UMinOp : LLVM_BinarySameArgsIntrinsicOp<"umin">;

def LLVM_MemcpyOp : LLVM_ZeroResultIntrOp<"memcpy", [0, 1, 2]> {
  let arguments = (ins LLVM_Type:$dst, LLVM_Type:$src, LLVM_Type:$len,
                   LLVM_Type:$isVolatile);
}
def LLVM_MemcpyInlineOp : LLVM_ZeroResultIntrOp<"memcpy.inline", [0, 1, 2]> {
  let arguments = (ins LLVM_Type:$dst, LLVM_Type:$src, LLVM_Type:$len,
                   LLVM_Type:$isVolatile);
}
def LLVM_MemmoveOp : LLVM_ZeroResultIntrOp<"memmove", [0, 1, 2]> {
  let arguments = (ins LLVM_Type:$dst, LLVM_Type:$src, LLVM_Type:$len,
                   LLVM_Type:$isVolatile);
}

def LLVM_MemsetOp : LLVM_ZeroResultIntrOp<"memset", [0, 2]> {
  let arguments = (ins LLVM_Type:$dst, LLVM_Type:$val, LLVM_Type:$len,
                   LLVM_Type:$isVolatile);
}

// Intrinsics with multiple returns.

def LLVM_SAddWithOverflowOp
    : LLVM_IntrOp<"sadd.with.overflow", [0], [], [], 2> {
  let arguments = (ins LLVM_Type, LLVM_Type);
}
def LLVM_UAddWithOverflowOp
    : LLVM_IntrOp<"uadd.with.overflow", [0], [], [], 2> {
  let arguments = (ins LLVM_Type, LLVM_Type);
}
def LLVM_SSubWithOverflowOp
    : LLVM_IntrOp<"ssub.with.overflow", [0], [], [], 2> {
  let arguments = (ins LLVM_Type, LLVM_Type);
}
def LLVM_USubWithOverflowOp
    : LLVM_IntrOp<"usub.with.overflow", [0], [], [], 2> {
  let arguments = (ins LLVM_Type, LLVM_Type);
}
def LLVM_SMulWithOverflowOp
    : LLVM_IntrOp<"smul.with.overflow", [0], [], [], 2> {
  let arguments = (ins LLVM_Type, LLVM_Type);
}
def LLVM_UMulWithOverflowOp
    : LLVM_IntrOp<"umul.with.overflow", [0], [], [], 2> {
  let arguments = (ins LLVM_Type, LLVM_Type);
}

//
// Coroutine intrinsics.
//

def LLVM_CoroIdOp : LLVM_IntrOp<"coro.id", [], [], [], 1> {
  let arguments = (ins I32:$align,
                       LLVM_i8Ptr:$promise,
                       LLVM_i8Ptr:$coroaddr,
                       LLVM_i8Ptr:$fnaddrs);
  let assemblyFormat = "$align `,` $promise `,` $coroaddr `,` $fnaddrs"
    " attr-dict `:` type($res)";
}

def LLVM_CoroBeginOp : LLVM_IntrOp<"coro.begin", [], [], [], 1> {
  let arguments = (ins LLVM_TokenType:$token,
                       LLVM_i8Ptr:$mem);
  let assemblyFormat = "$token `,` $mem attr-dict `:` type($res)";
}

def LLVM_CoroSizeOp : LLVM_IntrOp<"coro.size", [0], [], [], 1> {
  let assemblyFormat = "attr-dict `:` type($res)";
}

def LLVM_CoroAlignOp : LLVM_IntrOp<"coro.align", [0], [], [], 1> {
  let assemblyFormat = "attr-dict `:` type($res)";
}

def LLVM_CoroSaveOp : LLVM_IntrOp<"coro.save", [], [], [], 1> {
  let arguments = (ins LLVM_i8Ptr:$handle);
  let assemblyFormat = "$handle attr-dict `:` type($res)";
}

def LLVM_CoroSuspendOp : LLVM_IntrOp<"coro.suspend", [], [], [], 1> {
  let arguments = (ins LLVM_TokenType:$save,
                       I1:$final);
  let assemblyFormat = "$save `,` $final attr-dict `:` type($res)";
}

def LLVM_CoroEndOp : LLVM_IntrOp<"coro.end", [], [], [], 1> {
  let arguments = (ins LLVM_i8Ptr:$handle,
                       I1:$unwind);
  let assemblyFormat = "$handle `,` $unwind attr-dict `:` type($res)";
}

def LLVM_CoroFreeOp : LLVM_IntrOp<"coro.free", [], [], [], 1> {
  let arguments = (ins LLVM_TokenType:$id,
                       LLVM_i8Ptr:$handle);
  let assemblyFormat = "$id `,` $handle attr-dict `:` type($res)";
}

def LLVM_CoroResumeOp : LLVM_IntrOp<"coro.resume", [], [], [], 0> {
  let arguments = (ins LLVM_i8Ptr:$handle);
  let assemblyFormat = "$handle attr-dict";
}

//
// Exception handling intrinsics.
//

def LLVM_EhTypeidForOp : LLVM_OneResultIntrOp<"eh.typeid.for"> {
    let arguments = (ins LLVM_i8Ptr:$type_info);
    let assemblyFormat = "$type_info attr-dict `:` type($res)";
}

//
// Stack save/restore intrinsics.
//

def LLVM_StackSaveOp : LLVM_OneResultIntrOp<"stacksave"> {
  let assemblyFormat = "attr-dict `:` type($res)";
}

def LLVM_StackRestoreOp : LLVM_ZeroResultIntrOp<"stackrestore"> {
  let arguments = (ins LLVM_i8Ptr:$ptr);
  let assemblyFormat = "$ptr attr-dict";
}

//
// Vector Reductions.
//

def LLVM_vector_reduce_add : LLVM_VectorReduction<"add">;
def LLVM_vector_reduce_and : LLVM_VectorReduction<"and">;
def LLVM_vector_reduce_mul : LLVM_VectorReduction<"mul">;
def LLVM_vector_reduce_fmax : LLVM_VectorReduction<"fmax">;
def LLVM_vector_reduce_fmin : LLVM_VectorReduction<"fmin">;
def LLVM_vector_reduce_or : LLVM_VectorReduction<"or">;
def LLVM_vector_reduce_smax : LLVM_VectorReduction<"smax">;
def LLVM_vector_reduce_smin : LLVM_VectorReduction<"smin">;
def LLVM_vector_reduce_umax : LLVM_VectorReduction<"umax">;
def LLVM_vector_reduce_umin : LLVM_VectorReduction<"umin">;
def LLVM_vector_reduce_xor : LLVM_VectorReduction<"xor">;

def LLVM_vector_reduce_fadd : LLVM_VectorReductionAcc<"fadd">;
def LLVM_vector_reduce_fmul : LLVM_VectorReductionAcc<"fmul">;

//
// LLVM Matrix operations.
//

/// Create a column major, strided 2-D matrix load, as specified in the LLVM
/// MatrixBuilder.
/// data       - Start address of the matrix read
/// rows       - Number of rows in matrix (must be a constant)
/// isVolatile - True if the load operation is marked as volatile.
/// columns    - Number of columns in matrix (must be a constant)
/// stride     - Space between columns
def LLVM_MatrixColumnMajorLoadOp : LLVM_Op<"intr.matrix.column.major.load"> {
  let arguments = (ins LLVM_Type:$data, LLVM_Type:$stride, I1Attr:$isVolatile,
                   I32Attr:$rows, I32Attr:$columns);
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  string llvmBuilder = [{
    llvm::MatrixBuilder mb(builder);
    const llvm::DataLayout &dl =
      builder.GetInsertBlock()->getModule()->getDataLayout();
    llvm::Type *ElemTy = $data->getType()->getPointerElementType();
    llvm::Align align = dl.getABITypeAlign(ElemTy);
    $res = mb.CreateColumnMajorLoad(
      ElemTy, $data, align, $stride, $isVolatile, $rows,
      $columns);
  }];
  let assemblyFormat = "$data `,` `<` `stride` `=` $stride `>` attr-dict"
    "`:` type($res) `from` type($data) `stride` type($stride)";
}

/// Create a column major, strided 2-D matrix store, as specified in the LLVM
/// MatrixBuilder.
/// matrix     - Matrix to store
/// ptr        - Pointer to write back to
/// isVolatile - True if the load operation is marked as volatile.
/// rows       - Number of rows in matrix (must be a constant)
/// columns    - Number of columns in matrix (must be a constant)
/// stride     - Space between columns
def LLVM_MatrixColumnMajorStoreOp : LLVM_Op<"intr.matrix.column.major.store"> {
  let arguments = (ins LLVM_Type:$matrix, LLVM_Type:$data, LLVM_Type:$stride,
                   I1Attr:$isVolatile, I32Attr:$rows, I32Attr:$columns);
  let builders = [LLVM_VoidResultTypeOpBuilder, LLVM_ZeroResultOpBuilder];
  string llvmBuilder = [{
    llvm::MatrixBuilder mb(builder);
    const llvm::DataLayout &dl =
      builder.GetInsertBlock()->getModule()->getDataLayout();
    llvm::Align align = dl.getABITypeAlign(
      $data->getType()->getPointerElementType());
    mb.CreateColumnMajorStore(
      $matrix, $data, align, $stride, $isVolatile,
      $rows, $columns);
  }];
  let assemblyFormat = "$matrix `,` $data `,` `<` `stride` `=` $stride `>` "
    "attr-dict`:` type($matrix) `to` type($data) `stride` type($stride)";
}

/// Create a llvm.matrix.multiply call, multiplying 2-D matrices LHS and RHS, as
/// specified in the LLVM MatrixBuilder.
def LLVM_MatrixMultiplyOp : LLVM_Op<"intr.matrix.multiply"> {
  let arguments = (ins LLVM_Type:$lhs, LLVM_Type:$rhs, I32Attr:$lhs_rows,
                   I32Attr:$lhs_columns, I32Attr:$rhs_columns);
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  string llvmBuilder = [{
    llvm::MatrixBuilder mb(builder);
    $res = mb.CreateMatrixMultiply(
      $lhs, $rhs, $lhs_rows, $lhs_columns,
      $rhs_columns);
  }];
  let assemblyFormat = "$lhs `,` $rhs attr-dict "
    "`:` `(` type($lhs) `,` type($rhs) `)` `->` type($res)";
}

/// Create a llvm.matrix.transpose call, transposing a `rows` x `columns` 2-D
/// `matrix`, as specified in the LLVM MatrixBuilder.
def LLVM_MatrixTransposeOp : LLVM_Op<"intr.matrix.transpose"> {
  let arguments = (ins LLVM_Type:$matrix, I32Attr:$rows, I32Attr:$columns);
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  string llvmBuilder = [{
    llvm::MatrixBuilder mb(builder);
    $res = mb.CreateMatrixTranspose(
      $matrix, $rows, $columns);
  }];
  let assemblyFormat = "$matrix attr-dict `:` type($matrix) `into` type($res)";
}

//
// LLVM masked operations.
//

/// Create a llvm.get.active.lane.mask to set a mask up to a given position.
def LLVM_GetActiveLaneMaskOp
    : LLVM_OneResultIntrOp<"get.active.lane.mask", [0], [0], [NoSideEffect]> {
  let arguments = (ins LLVM_Type:$base, LLVM_Type:$n);
  let assemblyFormat = "$base `,` $n attr-dict `:` "
    "type($base) `,` type($n) `to` type($res)";
}

/// Create a call to Masked Load intrinsic.
def LLVM_MaskedLoadOp : LLVM_Op<"intr.masked.load"> {
  let arguments = (ins LLVM_Type:$data, LLVM_Type:$mask,
                   Variadic<LLVM_Type>:$pass_thru, I32Attr:$alignment);
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  string llvmBuilder = [{
    llvm::Type *Ty = $data->getType()->getPointerElementType();
    $res = $pass_thru.empty() ? builder.CreateMaskedLoad(
      Ty, $data, llvm::Align($alignment), $mask) :
      builder.CreateMaskedLoad(
        Ty, $data, llvm::Align($alignment), $mask, $pass_thru[0]);
  }];
  let assemblyFormat =
    "operands attr-dict `:` functional-type(operands, results)";
}

/// Create a call to Masked Store intrinsic.
def LLVM_MaskedStoreOp : LLVM_Op<"intr.masked.store"> {
  let arguments = (ins LLVM_Type:$value, LLVM_Type:$data, LLVM_Type:$mask,
                   I32Attr:$alignment);
  let builders = [LLVM_VoidResultTypeOpBuilder, LLVM_ZeroResultOpBuilder];
  string llvmBuilder = [{
    builder.CreateMaskedStore(
      $value, $data, llvm::Align($alignment), $mask);
  }];
  let assemblyFormat = "$value `,` $data `,` $mask attr-dict `:` "
    "type($value) `,` type($mask) `into` type($data)";
}

/// Create a call to Masked Gather intrinsic.
def LLVM_masked_gather : LLVM_Op<"intr.masked.gather"> {
  let arguments = (ins LLVM_Type:$ptrs, LLVM_Type:$mask,
                   Variadic<LLVM_Type>:$pass_thru, I32Attr:$alignment);
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  string llvmBuilder = [{
    llvm::VectorType *PtrVecTy = cast<llvm::VectorType>($ptrs->getType());
    llvm::Type *Ty = llvm::VectorType::get(
      PtrVecTy->getElementType()->getPointerElementType(),
      PtrVecTy->getElementCount());
    $res = $pass_thru.empty() ? builder.CreateMaskedGather(
      Ty, $ptrs, llvm::Align($alignment), $mask) :
      builder.CreateMaskedGather(
        Ty, $ptrs, llvm::Align($alignment), $mask, $pass_thru[0]);
  }];
  let assemblyFormat =
    "operands attr-dict `:` functional-type(operands, results)";
}

/// Create a call to Masked Scatter intrinsic.
def LLVM_masked_scatter : LLVM_Op<"intr.masked.scatter"> {
  let arguments = (ins LLVM_Type:$value, LLVM_Type:$ptrs, LLVM_Type:$mask,
                   I32Attr:$alignment);
  let builders = [LLVM_VoidResultTypeOpBuilder, LLVM_ZeroResultOpBuilder];
  string llvmBuilder = [{
    builder.CreateMaskedScatter(
      $value, $ptrs, llvm::Align($alignment), $mask);
  }];
  let assemblyFormat = "$value `,` $ptrs `,` $mask attr-dict `:` "
    "type($value) `,` type($mask) `into` type($ptrs)";
}

/// Create a call to Masked Expand Load intrinsic.
def LLVM_masked_expandload : LLVM_IntrOp<"masked.expandload", [0], [], [], 1> {
  let arguments = (ins LLVM_Type, LLVM_Type, LLVM_Type);
}

/// Create a call to Masked Compress Store intrinsic.
def LLVM_masked_compressstore
    : LLVM_IntrOp<"masked.compressstore", [], [0], [], 0> {
  let arguments = (ins LLVM_Type, LLVM_Type, LLVM_Type);
}

/// Create a call to vscale intrinsic.
def LLVM_vscale : LLVM_IntrOp<"vscale", [0], [], [], 1>;

// Atomic operations.
//

def AtomicBinOpXchg : I64EnumAttrCase<"xchg", 0>;
def AtomicBinOpAdd  : I64EnumAttrCase<"add", 1>;
def AtomicBinOpSub  : I64EnumAttrCase<"sub", 2>;
def AtomicBinOpAnd  : I64EnumAttrCase<"_and", 3>;
def AtomicBinOpNand : I64EnumAttrCase<"nand", 4>;
def AtomicBinOpOr   : I64EnumAttrCase<"_or", 5>;
def AtomicBinOpXor  : I64EnumAttrCase<"_xor", 6>;
def AtomicBinOpMax  : I64EnumAttrCase<"max", 7>;
def AtomicBinOpMin  : I64EnumAttrCase<"min", 8>;
def AtomicBinOpUMax : I64EnumAttrCase<"umax", 9>;
def AtomicBinOpUMin : I64EnumAttrCase<"umin", 10>;
def AtomicBinOpFAdd : I64EnumAttrCase<"fadd", 11>;
def AtomicBinOpFSub : I64EnumAttrCase<"fsub", 12>;
def AtomicBinOp : I64EnumAttr<
    "AtomicBinOp",
    "llvm.atomicrmw binary operations",
    [AtomicBinOpXchg, AtomicBinOpAdd, AtomicBinOpSub, AtomicBinOpAnd,
     AtomicBinOpNand, AtomicBinOpOr, AtomicBinOpXor, AtomicBinOpMax,
     AtomicBinOpMin, AtomicBinOpUMax, AtomicBinOpUMin, AtomicBinOpFAdd,
     AtomicBinOpFSub]> {
  let cppNamespace = "::mlir::LLVM";
}

def AtomicOrderingNotAtomic              : I64EnumAttrCase<"not_atomic", 0>;
def AtomicOrderingUnordered              : I64EnumAttrCase<"unordered", 1>;
def AtomicOrderingMonotonic              : I64EnumAttrCase<"monotonic", 2>;
def AtomicOrderingAcquire                : I64EnumAttrCase<"acquire", 4>;
def AtomicOrderingRelease                : I64EnumAttrCase<"release", 5>;
def AtomicOrderingAcquireRelease         : I64EnumAttrCase<"acq_rel", 6>;
def AtomicOrderingSequentiallyConsistent : I64EnumAttrCase<"seq_cst", 7>;
def AtomicOrdering : I64EnumAttr<
    "AtomicOrdering",
    "Atomic ordering for LLVM's memory model",
    [AtomicOrderingNotAtomic, AtomicOrderingUnordered, AtomicOrderingMonotonic,
     AtomicOrderingAcquire, AtomicOrderingRelease, AtomicOrderingAcquireRelease,
     AtomicOrderingSequentiallyConsistent]> {
  let cppNamespace = "::mlir::LLVM";
}

def LLVM_AtomicRMWType : AnyTypeOf<[LLVM_AnyFloat, AnyInteger]>;

// FIXME: Need to add alignment attribute to MLIR atomicrmw operation.
def LLVM_AtomicRMWOp : LLVM_Op<"atomicrmw"> {
  let arguments = (ins AtomicBinOp:$bin_op,
                   LLVM_PointerTo<LLVM_AtomicRMWType>:$ptr,
                   LLVM_AtomicRMWType:$val, AtomicOrdering:$ordering);
  let results = (outs LLVM_AtomicRMWType:$res);
  let llvmBuilder = [{
    $res = builder.CreateAtomicRMW(getLLVMAtomicBinOp($bin_op), $ptr, $val,
                                   llvm::MaybeAlign(),
                                   getLLVMAtomicOrdering($ordering));
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LLVM_AtomicCmpXchgType : AnyTypeOf<[AnyInteger, LLVM_AnyPointer]>;
def LLVM_AtomicCmpXchgResultType : Type<And<[
  LLVM_AnyStruct.predicate,
  CPred<"$_self.cast<::mlir::LLVM::LLVMStructType>().getBody().size() == 2">,
  SubstLeaves<"$_self",
              "$_self.cast<::mlir::LLVM::LLVMStructType>().getBody()[0]",
              LLVM_AtomicCmpXchgType.predicate>,
  SubstLeaves<"$_self",
              "$_self.cast<::mlir::LLVM::LLVMStructType>().getBody()[1]",
              I1.predicate>]>,
 "an LLVM struct type with any integer or pointer followed by a single-bit "
 "integer">;

// FIXME: Need to add alignment attribute to MLIR cmpxchg operation.
def LLVM_AtomicCmpXchgOp : LLVM_Op<"cmpxchg"> {
  let arguments = (ins LLVM_PointerTo<LLVM_AtomicCmpXchgType>:$ptr,
                   LLVM_AtomicCmpXchgType:$cmp, LLVM_AtomicCmpXchgType:$val,
                   AtomicOrdering:$success_ordering,
                   AtomicOrdering:$failure_ordering);
  let results = (outs LLVM_AtomicCmpXchgResultType:$res);
  let llvmBuilder = [{
    $res = builder.CreateAtomicCmpXchg($ptr, $cmp, $val, llvm::MaybeAlign(),
                   getLLVMAtomicOrdering($success_ordering),
                   getLLVMAtomicOrdering($failure_ordering));
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LLVM_AssumeOp : LLVM_Op<"intr.assume", []> {
  let arguments = (ins LLVM_Type:$cond);
  let llvmBuilder = [{
    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn =
        llvm::Intrinsic::getDeclaration(module, llvm::Intrinsic::assume, {});
    builder.CreateCall(fn, {$cond});
  }];
}

def LLVM_FenceOp : LLVM_Op<"fence"> {
  let arguments = (ins AtomicOrdering:$ordering, StrAttr:$syncscope);
  let builders = [LLVM_VoidResultTypeOpBuilder, LLVM_ZeroResultOpBuilder];
  let llvmBuilder = [{
    llvm::LLVMContext &llvmContext = builder.getContext();
    builder.CreateFence(getLLVMAtomicOrdering($ordering),
      llvmContext.getOrInsertSyncScopeID($syncscope));
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def AsmATT : LLVM_EnumAttrCase<
  /*string cppSym=*/"AD_ATT", /*string irSym=*/"att",
  /*string llvmSym=*/"AD_ATT", /*int val=*/0>;
def AsmIntel : LLVM_EnumAttrCase<
  /*string cppSym=*/"AD_Intel", /*string irSym=*/"intel",
  /*string llvmSym=*/"AD_Intel", /*int val=*/1>;
def AsmATTOrIntel : LLVM_EnumAttr<
  /*string name=*/"AsmDialect",
  /*string llvmName=*/"::llvm::InlineAsm::AsmDialect",
  /*string description=*/"ATT (0) or Intel (1) asm dialect",
  /*list<LLVM_EnumAttrCase> cases=*/[AsmATT, AsmIntel]> {
  let cppNamespace = "::mlir::LLVM";
}

def LLVM_InlineAsmOp : LLVM_Op<"inline_asm", []> {
  let description = [{
    The InlineAsmOp mirrors the underlying LLVM semantics with a notable
    exception: the embedded `asm_string` is not allowed to define or reference
    any symbol or any global variable: only the operands of the op may be read,
    written, or referenced.
    Attempting to define or reference any symbol or any global behavior is
    considered undefined behavior at this time.
  }];
  let arguments = (
    ins Variadic<LLVM_Type>:$operands,
        StrAttr:$asm_string,
        StrAttr:$constraints,
        UnitAttr:$has_side_effects,
        UnitAttr:$is_align_stack,
        OptionalAttr<
          DefaultValuedAttr<AsmATTOrIntel, "AsmDialect::AD_ATT">>:$asm_dialect,
        OptionalAttr<ArrayAttr>:$operand_attrs);

  let results = (outs Optional<LLVM_Type>:$res);

  let assemblyFormat = [{
    (`has_side_effects` $has_side_effects^)?
    (`is_align_stack` $is_align_stack^)?
    (`asm_dialect` `=` $asm_dialect^)?
    (`operand_attrs` `=` $operand_attrs^)?
    attr-dict
    $asm_string `,` $constraints
    operands `:` functional-type(operands, results)
   }];

  let extraClassDeclaration = [{
    static StringRef getElementTypeAttrName() {
      return "elementtype";
    }
  }];
}
#endif // LLVMIR_OPS
