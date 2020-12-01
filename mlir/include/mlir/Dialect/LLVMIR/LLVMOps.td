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
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class LLVM_Builder<string builder> {
  string llvmBuilder = builder;
}

def LLVM_OneResultOpBuilder :
  OpBuilderDAG<(ins "Type":$resultType, "ValueRange":$operands,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    if (resultType) $_state.addTypes(resultType);
    $_state.addOperands(operands);
    for (auto namedAttr : attributes) {
      $_state.addAttribute(namedAttr.first, namedAttr.second);
    }
  }]>;

def LLVM_ZeroResultOpBuilder :
  OpBuilderDAG<(ins "ValueRange":$operands,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(operands);
    for (auto namedAttr : attributes) {
      $_state.addAttribute(namedAttr.first, namedAttr.second);
    }
  }]>;

class LLVM_TwoBuilders<OpBuilderDAG b1, OpBuilderDAG b2> {
  list<OpBuilderDAG> builders = [b1, b2];
}

// Base class for LLVM operations with one result.
class LLVM_OneResultOp<string mnemonic, list<OpTrait> traits = []> :
    LLVM_Op<mnemonic, traits>, Results<(outs LLVM_Type:$res)> {
  let builders = [LLVM_OneResultOpBuilder];
}

// Compatibility builder that takes an instance of wrapped llvm::VoidType
// to indicate no result.
def LLVM_VoidResultTypeOpBuilder :
  OpBuilderDAG<(ins "Type":$resultType, "ValueRange":$operands,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    auto llvmType = resultType.dyn_cast<LLVMType>(); (void)llvmType;
    assert(llvmType && "result must be an LLVM type");
    assert(llvmType.isVoidTy() &&
           "for zero-result operands, only 'void' is accepted as result type");
    build($_builder, $_state, operands, attributes);
  }]>;

// Base class for LLVM operations with zero results.
class LLVM_ZeroResultOp<string mnemonic, list<OpTrait> traits = []> :
    LLVM_Op<mnemonic, traits>, Results<(outs)>,
    LLVM_TwoBuilders<LLVM_VoidResultTypeOpBuilder, LLVM_ZeroResultOpBuilder>;

// Opaque builder used for terminator operations that contain successors.
def LLVM_TerminatorPassthroughOpBuilder :
  OpBuilderDAG<(ins "ValueRange":$operands, "SuccessorRange":$destinations,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(operands);
    $_state.addSuccessors(destinations);
    $_state.addAttributes(attributes);
  }]>;

// Base class for LLVM terminator operations.  All terminator operations have
// zero results and an optional list of successors.
class LLVM_TerminatorOp<string mnemonic, list<OpTrait> traits = []> :
    LLVM_Op<mnemonic, !listconcat(traits, [Terminator])>;

// Class for arithmetic binary operations.
class LLVM_ArithmeticOpBase<Type type, string mnemonic,
                            string builderFunc, list<OpTrait> traits = []> :
    LLVM_OneResultOp<mnemonic,
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)>,
    LLVM_Builder<"$res = builder." # builderFunc # "($lhs, $rhs);"> {
  let arguments = (ins LLVM_ScalarOrVectorOf<type>:$lhs,
                   LLVM_ScalarOrVectorOf<type>:$rhs);
  let parser =
      [{ return impl::parseOneResultSameOperandTypeOp(parser, result); }];
  let printer = [{ mlir::impl::printOneResultOp(this->getOperation(), p); }];
}
class LLVM_IntArithmeticOp<string mnemonic, string builderFunc,
                           list<OpTrait> traits = []> :
    LLVM_ArithmeticOpBase<LLVM_AnyInteger, mnemonic, builderFunc, traits>;
class LLVM_FloatArithmeticOp<string mnemonic, string builderFunc,
                             list<OpTrait> traits = []> :
    LLVM_ArithmeticOpBase<LLVM_AnyFloat, mnemonic, builderFunc, traits>;

// Class for arithmetic unary operations.
class LLVM_UnaryArithmeticOp<Type type, string mnemonic,
                             string builderFunc, list<OpTrait> traits = []> :
    LLVM_OneResultOp<mnemonic,
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)>,
    LLVM_Builder<"$res = builder." # builderFunc # "($operand);"> {
  let arguments = (ins type:$operand);
  let parser =
      [{ return impl::parseOneResultSameOperandTypeOp(parser, result); }];
  let printer = [{ mlir::impl::printOneResultOp(this->getOperation(), p); }];
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
def LLVM_ICmpOp : LLVM_OneResultOp<"icmp", [NoSideEffect]> {
  let arguments = (ins ICmpPredicate:$predicate,
                   LLVM_ScalarOrVectorOf<LLVM_AnyInteger>:$lhs,
                   LLVM_ScalarOrVectorOf<LLVM_AnyInteger>:$rhs);
  let llvmBuilder = [{
    $res = builder.CreateICmp(getLLVMCmpPredicate($predicate), $lhs, $rhs);
  }];
  let builders = [
    OpBuilderDAG<(ins "ICmpPredicate":$predicate, "Value":$lhs, "Value":$rhs),
    [{
      build($_builder, $_state, LLVMType::getInt1Ty(lhs.getType().getContext()),
            $_builder.getI64IntegerAttr(static_cast<int64_t>(predicate)), lhs, rhs);
    }]>];
  let parser = [{ return parseCmpOp<ICmpPredicate>(parser, result); }];
  let printer = [{ printICmpOp(p, *this); }];
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

// Other integer operations.
def LLVM_FCmpOp : LLVM_OneResultOp<"fcmp", [NoSideEffect]> {
  let arguments = (ins FCmpPredicate:$predicate,
                   LLVM_ScalarOrVectorOf<LLVM_AnyFloat>:$lhs,
                   LLVM_ScalarOrVectorOf<LLVM_AnyFloat>:$rhs);
  let llvmBuilder = [{
    $res = builder.CreateFCmp(getLLVMCmpPredicate($predicate), $lhs, $rhs);
  }];
  let builders = [
    OpBuilderDAG<(ins "FCmpPredicate":$predicate, "Value":$lhs, "Value":$rhs),
    [{
      build($_builder, $_state, LLVMType::getInt1Ty(lhs.getType().getContext()),
            $_builder.getI64IntegerAttr(static_cast<int64_t>(predicate)), lhs, rhs);
    }]>];
  let parser = [{ return parseCmpOp<FCmpPredicate>(parser, result); }];
  let printer = [{ printFCmpOp(p, *this); }];
}

// Floating point binary operations.
def LLVM_FAddOp : LLVM_FloatArithmeticOp<"fadd", "CreateFAdd">;
def LLVM_FSubOp : LLVM_FloatArithmeticOp<"fsub", "CreateFSub">;
def LLVM_FMulOp : LLVM_FloatArithmeticOp<"fmul", "CreateFMul">;
def LLVM_FDivOp : LLVM_FloatArithmeticOp<"fdiv", "CreateFDiv">;
def LLVM_FRemOp : LLVM_FloatArithmeticOp<"frem", "CreateFRem">;
def LLVM_FNegOp : LLVM_UnaryArithmeticOp<LLVM_ScalarOrVectorOf<LLVM_AnyFloat>,
                                         "fneg", "CreateFNeg">;

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
}

// Memory-related operations.
def LLVM_AllocaOp :
    MemoryOpWithAlignmentBase,
    LLVM_OneResultOp<"alloca"> {
  let arguments = (ins LLVM_AnyInteger:$arraySize,
                   OptionalAttr<I64Attr>:$alignment);
  string llvmBuilder = [{
    auto *inst = builder.CreateAlloca(
      $_resultType->getPointerElementType(), $arraySize);
    }] # setAlignmentCode # [{
    $res = inst;
  }];
  let builders = [
    OpBuilderDAG<(ins "Type":$resultType, "Value":$arraySize,
      "unsigned":$alignment),
    [{
      if (alignment == 0)
        return build($_builder, $_state, resultType, arraySize, IntegerAttr());
      build($_builder, $_state, resultType, arraySize,
        $_builder.getI64IntegerAttr(alignment));
  }]>];
  let parser = [{ return parseAllocaOp(parser, result); }];
  let printer = [{ printAllocaOp(p, *this); }];
}

def LLVM_GEPOp : LLVM_OneResultOp<"getelementptr", [NoSideEffect]>,
                 LLVM_Builder<"$res = builder.CreateGEP($base, $indices);"> {
  let arguments = (ins LLVM_ScalarOrVectorOf<LLVM_AnyPointer>:$base,
                   Variadic<LLVM_ScalarOrVectorOf<LLVM_AnyInteger>>:$indices);
  let assemblyFormat = [{
    $base `[` $indices `]` attr-dict `:` functional-type(operands, results)
  }];
}

def LLVM_LoadOp :
    MemoryOpWithAlignmentAndAttributes,
    LLVM_OneResultOp<"load"> {
  let arguments = (ins LLVM_PointerTo<LLVM_LoadableType>:$addr,
                   OptionalAttr<I64Attr>:$alignment, UnitAttr:$volatile_,
                   UnitAttr:$nontemporal);
  string llvmBuilder = [{
    auto *inst = builder.CreateLoad($addr, $volatile_);
  }] # setAlignmentCode # setNonTemporalMetadataCode # [{
    $res = inst;
  }];
  let builders = [
    OpBuilderDAG<(ins "Value":$addr, CArg<"unsigned", "0">:$alignment,
      CArg<"bool", "false">:$isVolatile, CArg<"bool", "false">:$isNonTemporal),
    [{
      auto type = addr.getType().cast<LLVMType>().getPointerElementTy();
      build($_builder, $_state, type, addr, alignment, isVolatile, isNonTemporal);
    }]>,
    OpBuilderDAG<(ins "Type":$t, "Value":$addr,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isVolatile,
      CArg<"bool", "false">:$isNonTemporal)>];
  let parser = [{ return parseLoadOp(parser, result); }];
  let printer = [{ printLoadOp(p, *this); }];
}
def LLVM_StoreOp :
    MemoryOpWithAlignmentAndAttributes,
    LLVM_ZeroResultOp<"store"> {
  let arguments = (ins LLVM_LoadableType:$value,
                   LLVM_PointerTo<LLVM_LoadableType>:$addr,
                   OptionalAttr<I64Attr>:$alignment, UnitAttr:$volatile_,
                   UnitAttr:$nontemporal);
  string llvmBuilder = [{
    auto *inst = builder.CreateStore($value, $addr, $volatile_);
  }] # setAlignmentCode # setNonTemporalMetadataCode;
  let builders = [
    OpBuilderDAG<(ins "Value":$value, "Value":$addr,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isVolatile,
      CArg<"bool", "false">:$isNonTemporal)>
    ];
  let parser = [{ return parseStoreOp(parser, result); }];
  let printer = [{ printStoreOp(p, *this); }];
}

// Casts.
class LLVM_CastOp<string mnemonic, string builderFunc, Type type,
                  list<OpTrait> traits = []> :
    LLVM_OneResultOp<mnemonic,
           !listconcat([NoSideEffect], traits)>,
    LLVM_Builder<"$res = builder." # builderFunc # "($arg, $_resultType);"> {
  let arguments = (ins type:$arg);
  let parser = [{ return mlir::impl::parseCastOp(parser, result); }];
  let printer = [{ mlir::impl::printCastOp(this->getOperation(), p); }];
}
def LLVM_BitcastOp : LLVM_CastOp<"bitcast", "CreateBitCast",
                                 LLVM_AnyNonAggregate>;
def LLVM_AddrSpaceCastOp : LLVM_CastOp<"addrspacecast", "CreateAddrSpaceCast",
                                       LLVM_ScalarOrVectorOf<LLVM_AnyPointer>>;
def LLVM_IntToPtrOp : LLVM_CastOp<"inttoptr", "CreateIntToPtr",
                                  LLVM_ScalarOrVectorOf<LLVM_AnyInteger>>;
def LLVM_PtrToIntOp : LLVM_CastOp<"ptrtoint", "CreatePtrToInt",
                                  LLVM_ScalarOrVectorOf<LLVM_AnyPointer>>;
def LLVM_SExtOp : LLVM_CastOp<"sext", "CreateSExt",
                              LLVM_ScalarOrVectorOf<LLVM_AnyInteger>>;
def LLVM_ZExtOp : LLVM_CastOp<"zext", "CreateZExt",
                              LLVM_ScalarOrVectorOf<LLVM_AnyInteger>>;
def LLVM_TruncOp : LLVM_CastOp<"trunc", "CreateTrunc",
                               LLVM_ScalarOrVectorOf<LLVM_AnyInteger>>;
def LLVM_SIToFPOp : LLVM_CastOp<"sitofp", "CreateSIToFP",
                                LLVM_ScalarOrVectorOf<LLVM_AnyInteger>>;
def LLVM_UIToFPOp : LLVM_CastOp<"uitofp", "CreateUIToFP",
                                LLVM_ScalarOrVectorOf<LLVM_AnyInteger>>;
def LLVM_FPToSIOp : LLVM_CastOp<"fptosi", "CreateFPToSI",
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>>;
def LLVM_FPToUIOp : LLVM_CastOp<"fptoui", "CreateFPToUI",
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>>;
def LLVM_FPExtOp : LLVM_CastOp<"fpext", "CreateFPExt",
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>>;
def LLVM_FPTruncOp : LLVM_CastOp<"fptrunc", "CreateFPTrunc",
                                 LLVM_ScalarOrVectorOf<LLVM_AnyFloat>>;

// Call-related operations.
def LLVM_InvokeOp : LLVM_Op<"invoke", [
                      AttrSizedOperandSegments,
                      DeclareOpInterfaceMethods<BranchOpInterface>,
                      Terminator
                    ]>,
                    Results<(outs Variadic<LLVM_Type>)> {
  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$callee,
                   Variadic<LLVM_Type>:$operands,
                   Variadic<LLVM_Type>:$normalDestOperands,
                   Variadic<LLVM_Type>:$unwindDestOperands);
  let successors = (successor AnySuccessor:$normalDest,
                              AnySuccessor:$unwindDest);

  let builders = [
    OpBuilderDAG<(ins "TypeRange":$tys, "FlatSymbolRefAttr":$callee,
      "ValueRange":$ops, "Block*":$normal, "ValueRange":$normalOps,
      "Block*":$unwind, "ValueRange":$unwindOps),
    [{
      $_state.addAttribute("callee", callee);
      build($_builder, $_state, tys, ops, normal, normalOps, unwind, unwindOps);
    }]>,
    OpBuilderDAG<(ins "TypeRange":$tys, "ValueRange":$ops, "Block*":$normal,
      "ValueRange":$normalOps, "Block*":$unwind, "ValueRange":$unwindOps),
    [{
      build($_builder, $_state, tys, /*callee=*/FlatSymbolRefAttr(), ops, normalOps,
            unwindOps, normal, unwind);
    }]>];
  let verifier = [{ return ::verify(*this);  }];
  let parser = [{ return parseInvokeOp(parser, result); }];
  let printer = [{ printInvokeOp(p, *this); }];
}

def LLVM_LandingpadOp : LLVM_OneResultOp<"landingpad"> {
  let arguments = (ins UnitAttr:$cleanup, Variadic<LLVM_Type>);
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return parseLandingpadOp(parser, result); }];
  let printer = [{ printLandingpadOp(p, *this); }];
}

def LLVM_CallOp : LLVM_Op<"call">,
                  Results<(outs Variadic<LLVM_Type>)> {
  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$callee,
                   Variadic<LLVM_Type>);
  let builders = [
    OpBuilderDAG<(ins "LLVMFuncOp":$func, "ValueRange":$operands,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
    [{
      LLVMType resultType = func.getType().getFunctionResultType();
      if (!resultType.isVoidTy())
        $_state.addTypes(resultType);
      $_state.addAttribute("callee", $_builder.getSymbolRefAttr(func));
      $_state.addAttributes(attributes);
      $_state.addOperands(operands);
    }]>];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return parseCallOp(parser, result); }];
  let printer = [{ printCallOp(p, *this); }];
}
def LLVM_ExtractElementOp : LLVM_OneResultOp<"extractelement", [NoSideEffect]> {
  let arguments = (ins LLVM_AnyVector:$vector, LLVM_AnyInteger:$position);
  string llvmBuilder = [{
    $res = builder.CreateExtractElement($vector, $position);
  }];
  let builders = [
    OpBuilderDAG<(ins "Value":$vector, "Value":$position,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];
  let parser = [{ return parseExtractElementOp(parser, result); }];
  let printer = [{ printExtractElementOp(p, *this); }];
}
def LLVM_ExtractValueOp : LLVM_OneResultOp<"extractvalue", [NoSideEffect]> {
  let arguments = (ins LLVM_AnyAggregate:$container, ArrayAttr:$position);
  string llvmBuilder = [{
    $res = builder.CreateExtractValue($container, extractPosition($position));
  }];
  let parser = [{ return parseExtractValueOp(parser, result); }];
  let printer = [{ printExtractValueOp(p, *this); }];
}
def LLVM_InsertElementOp : LLVM_OneResultOp<"insertelement", [NoSideEffect]> {
  let arguments = (ins LLVM_AnyVector:$vector, LLVM_PrimitiveType:$value,
                   LLVM_AnyInteger:$position);
  string llvmBuilder = [{
    $res = builder.CreateInsertElement($vector, $value, $position);
  }];
  let parser = [{ return parseInsertElementOp(parser, result); }];
  let printer = [{ printInsertElementOp(p, *this); }];
}
def LLVM_InsertValueOp : LLVM_OneResultOp<"insertvalue", [NoSideEffect]> {
  let arguments = (ins LLVM_AnyAggregate:$container, LLVM_PrimitiveType:$value,
                   ArrayAttr:$position);
  string llvmBuilder = [{
    $res = builder.CreateInsertValue($container, $value,
                                     extractPosition($position));
  }];
  let builders = [
    OpBuilderDAG<(ins "Value":$container, "Value":$value, "ArrayAttr":$position),
    [{
      build($_builder, $_state, container.getType(), container, value, position);
    }]>];
  let parser = [{ return parseInsertValueOp(parser, result); }];
  let printer = [{ printInsertValueOp(p, *this); }];
}
def LLVM_ShuffleVectorOp : LLVM_OneResultOp<"shufflevector", [NoSideEffect]> {
  let arguments = (ins LLVM_AnyVector:$v1, LLVM_AnyVector:$v2, ArrayAttr:$mask);
  string llvmBuilder = [{
      SmallVector<unsigned, 4> position = extractPosition($mask);
      SmallVector<int, 4> mask(position.begin(), position.end());
      $res = builder.CreateShuffleVector($v1, $v2, mask);
  }];
  let builders = [
    OpBuilderDAG<(ins "Value":$v1, "Value":$v2, "ArrayAttr":$mask,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];
  let verifier = [{
    auto wrappedVectorType1 = v1().getType().cast<LLVMType>();
    auto wrappedVectorType2 = v2().getType().cast<LLVMType>();
    if (!wrappedVectorType2.isVectorTy())
      return emitOpError("expected LLVM IR Dialect vector type for operand #2");
    if (wrappedVectorType1.getVectorElementType() !=
        wrappedVectorType2.getVectorElementType())
      return emitOpError("expected matching LLVM IR Dialect element types");
    return success();
  }];
  let parser = [{ return parseShuffleVectorOp(parser, result); }];
  let printer = [{ printShuffleVectorOp(p, *this); }];
}

// Misc operations.
def LLVM_SelectOp
    : LLVM_OneResultOp<"select",
          [NoSideEffect, AllTypesMatch<["trueValue", "falseValue", "res"]>]>,
      LLVM_Builder<
          "$res = builder.CreateSelect($condition, $trueValue, $falseValue);"> {
  let arguments = (ins LLVM_ScalarOrVectorOf<LLVM_i1>:$condition,
                   LLVM_Type:$trueValue, LLVM_Type:$falseValue);
  let builders = [
    OpBuilderDAG<(ins "Value":$condition, "Value":$lhs, "Value":$rhs),
    [{
      build($_builder, $_state, lhs.getType(), condition, lhs, rhs);
    }]>];
  let assemblyFormat = "operands attr-dict `:` type($condition) `,` type($res)";
}
def LLVM_FreezeOp : LLVM_OneResultOp<"freeze", [SameOperandsAndResultType]> {
  let arguments = (ins LLVM_Type:$val);
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
  let arguments = (ins LLVM_i1:$condition,
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
    OpBuilderDAG<(ins "Value":$condition, "Block *":$trueDest,
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
  OpBuilderDAG<(ins "Value":$condition, "Block *":$trueDest,
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

  let verifier = [{
    if (getNumOperands() > 1)
      return emitOpError("expects at most 1 operand");
    return success();
  }];

  let parser = [{ return parseReturnOp(parser, result); }];
  let printer = [{ printReturnOp(p, *this); }];
}
def LLVM_ResumeOp : LLVM_TerminatorOp<"resume", []> {
  let arguments = (ins LLVM_Type:$value);
  string llvmBuilder = [{ builder.CreateResume($value); }];
  let verifier = [{
    if (!isa_and_nonnull<LandingpadOp>(value().getDefiningOp()))
      return emitOpError("expects landingpad value as operand");
    // No check for personality of function - landingpad op verifies it.
    return success();
  }];

  let assemblyFormat = "$value attr-dict `:` type($value)";
}
def LLVM_UnreachableOp : LLVM_TerminatorOp<"unreachable", []> {
  string llvmBuilder = [{ builder.CreateUnreachable(); }];
  let parser = [{ return success(); }];
  let printer = [{ p << getOperationName(); }];
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

def Linkage : LLVM_EnumAttr<
    "Linkage",
    "::llvm::GlobalValue::LinkageTypes",
    "LLVM linkage types",
    [LinkagePrivate, LinkageInternal, LinkageAvailableExternally,
     LinkageLinkonce, LinkageWeak, LinkageCommon, LinkageAppending,
     LinkageExternWeak, LinkageLinkonceODR, LinkageWeakODR, LinkageExternal]> {
  let cppNamespace = "::mlir::LLVM";
}


def LLVM_AddressOfOp : LLVM_OneResultOp<"mlir.addressof"> {
  let arguments = (ins FlatSymbolRefAttr:$global_name);

  let summary = "Creates a pointer pointing to a global or a function";

  let builders = [
    OpBuilderDAG<(ins "LLVMType":$resType, "StringRef":$name,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      $_state.addAttribute("global_name",$_builder.getSymbolRefAttr(name));
      $_state.addAttributes(attrs);
      $_state.addTypes(resType);}]>,
    OpBuilderDAG<(ins "GlobalOp":$global,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      build($_builder, $_state,
            global.getType().getPointerTo(global.addr_space()),
            global.sym_name(), attrs);}]>,
    OpBuilderDAG<(ins "LLVMFuncOp":$func,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      build($_builder, $_state,
            func.getType().getPointerTo(), func.getName(), attrs);}]>
  ];

  let extraClassDeclaration = [{
    /// Return the llvm.mlir.global operation that defined the value referenced
    /// here.
    GlobalOp getGlobal();

    /// Return the llvm.func operation that is referenced here.
    LLVMFuncOp getFunction();
  }];

  let assemblyFormat = "$global_name attr-dict `:` type($res)";
  let verifier = "return ::verify(*this);";
}

def LLVM_GlobalOp
    : LLVM_ZeroResultOp<"mlir.global",
                        [IsolatedFromAbove,
                         SingleBlockImplicitTerminator<"ReturnOp">, Symbol]> {
  let arguments = (ins
    TypeAttr:$type,
    UnitAttr:$constant,
    StrAttr:$sym_name,
    Linkage:$linkage,
    OptionalAttr<AnyAttr>:$value,
    DefaultValuedAttr<Confined<I32Attr, [IntNonNegative]>, "0">:$addr_space
  );
  let summary = "LLVM dialect global.";
  let description = [{
    Can contain an optional initializer region or attribute for simple
    initializers.

    Examples:
      // Initialized using an attribute.
      llvm.mlir.global @a("abc") : !llvm<"[3 x i8]">
      // Initialized using a region.
      llvm.mlir.global constant @b() : !llvm<"i32*"> {
        %0 = llvm.constant(0 : i32) : !llvm.i32
        %1 = llvm.inttoptr %0 : !llvm.i32 to !llvm<"i32*">
        llvm.return %1 : !llvm<"i32*">
      }
  }];
  let regions = (region AnyRegion:$initializer);

  let builders = [
    OpBuilderDAG<(ins "LLVMType":$type, "bool":$isConstant, "Linkage":$linkage,
      "StringRef":$name, "Attribute":$value, CArg<"unsigned", "0">:$addrSpace,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    /// Return the LLVM type of the global.
    LLVMType getType() {
      return type().cast<LLVMType>();
    }
    /// Return the initializer attribute if it exists, or a null attribute.
    Attribute getValueOrNull() {
      return value().getValueOr(Attribute());
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

  let printer = "printGlobalOp(p, *this);";
  let parser = "return parseGlobalOp(parser, result);";
  let verifier = "return ::verify(*this);";
}

def LLVM_LLVMFuncOp
    : LLVM_ZeroResultOp<"func", [AutomaticAllocationScope, IsolatedFromAbove,
                                 FunctionLike, Symbol]> {
  let summary = "LLVM dialect function, has wrapped LLVM IR function type";

  let arguments = (ins DefaultValuedAttr<Linkage, "Linkage::External">:$linkage,
                   OptionalAttr<FlatSymbolRefAttr>:$personality,
                   OptionalAttr<ArrayAttr>:$passthrough);

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilderDAG<(ins "StringRef":$name, "LLVMType":$type,
      CArg<"Linkage", "Linkage::External">:$linkage,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<MutableDictionaryAttr>", "{}">:$argAttrs)>
  ];

  let extraClassDeclaration = [{
    // Add an entry block to an empty function, and set up the block arguments
    // to match the signature of the function.
    Block *addEntryBlock();

    LLVMType getType() {
      return getAttrOfType<TypeAttr>(getTypeAttrName())
          .getValue().cast<LLVMType>();
    }
    bool isVarArg() {
      return getType().isFunctionVarArg();
    }

    // Hook for OpTrait::FunctionLike, returns the number of function arguments`.
    // Depends on the type attribute being correct as checked by verifyType.
    unsigned getNumFuncArguments();

    // Hook for OpTrait::FunctionLike, returns the number of function results.
    // Depends on the type attribute being correct as checked by verifyType.
    unsigned getNumFuncResults();

    // Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    // attribute is present.  This can check for preconditions of the
    // getNumArguments hook not failing.
    LogicalResult verifyType();
  }];

  let verifier = [{ return ::verify(*this); }];
  let printer = [{ printLLVMFuncOp(p, *this); }];
  let parser = [{ return parseLLVMFuncOp(parser, result); }];
}

def LLVM_NullOp
    : LLVM_OneResultOp<"mlir.null", [NoSideEffect]>,
      LLVM_Builder<"$res = llvm::ConstantPointerNull::get("
                   "    cast<llvm::PointerType>($_resultType));"> {
  let assemblyFormat = "attr-dict `:` type($res)";
  let verifier = [{ return ::verify(*this); }];
}

def LLVM_UndefOp : LLVM_OneResultOp<"mlir.undef", [NoSideEffect]>,
                   LLVM_Builder<"$res = llvm::UndefValue::get($_resultType);"> {
  let assemblyFormat = "attr-dict `:` type($res)";
}
def LLVM_ConstantOp
    : LLVM_OneResultOp<"mlir.constant", [NoSideEffect]>,
      LLVM_Builder<"$res = getLLVMConstant($_resultType, $value, $_location);">
{
  let arguments = (ins AnyAttr:$value);
  let assemblyFormat = "`(` $value `)` attr-dict `:` type($res)";
  let verifier = [{ return ::verify(*this); }];
}

def LLVM_DialectCastOp : LLVM_Op<"mlir.cast", [NoSideEffect]>,
                         Results<(outs AnyType:$res)> {
  let summary = "Type cast between LLVM dialect and Standard.";
  let description = [{
    llvm.mlir.cast op casts between Standard and LLVM dialects. It only changes
    the dialect, but does not change compile-time or runtime semantics.

    Notice that index type is not supported, as it's Standard-specific.

    Example:
      llvm.mlir.cast %v : f16 to llvm.half
      llvm.mlir.cast %v : llvm.float to f32
      llvm.mlir.cast %v : !llvm<"<2 x float>"> to vector<2xf32>
  }];
  let arguments = (ins AnyType:$in);
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($res)";
  let verifier = "return ::verify(*this);";
}

// Operations that correspond to LLVM intrinsics. With MLIR operation set being
// extendable, there is no reason to introduce a hard boundary between "core"
// operations and intrinsics. However, we systematically prefix them with
// "intr." to avoid potential name clashes.

class LLVM_UnaryIntrinsicOp<string func, list<OpTrait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)> {
  let arguments = (ins LLVM_Type:$in);
}

class LLVM_BinarySameArgsIntrinsicOp<string func, list<OpTrait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)> {
  let arguments = (ins LLVM_Type:$a, LLVM_Type:$b);
}

class LLVM_TernarySameArgsIntrinsicOp<string func, list<OpTrait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)> {
  let arguments = (ins LLVM_Type:$a, LLVM_Type:$b, LLVM_Type:$c);
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
def LLVM_BitReverseOp : LLVM_UnaryIntrinsicOp<"bitreverse">;
def LLVM_CtPopOp : LLVM_UnaryIntrinsicOp<"ctpop">;
def LLVM_MaxNumOp : LLVM_BinarySameArgsIntrinsicOp<"maxnum">;
def LLVM_MinNumOp : LLVM_BinarySameArgsIntrinsicOp<"minnum">;
def LLVM_SMaxOp : LLVM_BinarySameArgsIntrinsicOp<"smax">;
def LLVM_SMinOp : LLVM_BinarySameArgsIntrinsicOp<"smin">;

def LLVM_MemcpyOp : LLVM_ZeroResultIntrOp<"memcpy", [0, 1, 2]> {
  let arguments = (ins LLVM_Type:$dst, LLVM_Type:$src, LLVM_Type:$len,
                   LLVM_Type:$isVolatile);
}
def LLVM_MemcpyInlineOp : LLVM_ZeroResultIntrOp<"memcpy.inline", [0, 1, 2]> {
  let arguments = (ins LLVM_Type:$dst, LLVM_Type:$src, LLVM_Type:$len,
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
def LLVM_MatrixColumnMajorLoadOp
    : LLVM_OneResultOp<"intr.matrix.column.major.load"> {
  let arguments = (ins LLVM_Type:$data, LLVM_Type:$stride, I1Attr:$isVolatile,
                   I32Attr:$rows, I32Attr:$columns);
  string llvmBuilder = [{
    llvm::MatrixBuilder<decltype(builder)> mb(builder);
    const llvm::DataLayout &dl =
      builder.GetInsertBlock()->getModule()->getDataLayout();
    llvm::Align align = dl.getABITypeAlign(
      $data->getType()->getPointerElementType());
    $res = mb.CreateColumnMajorLoad(
      $data, align, $stride, $isVolatile, $rows,
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
def LLVM_MatrixColumnMajorStoreOp
    : LLVM_ZeroResultOp<"intr.matrix.column.major.store"> {
  let arguments = (ins LLVM_Type:$matrix, LLVM_Type:$data, LLVM_Type:$stride,
                   I1Attr:$isVolatile, I32Attr:$rows, I32Attr:$columns);
  string llvmBuilder = [{
    llvm::MatrixBuilder<decltype(builder)> mb(builder);
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
def LLVM_MatrixMultiplyOp
    : LLVM_OneResultOp<"intr.matrix.multiply"> {
  let arguments = (ins LLVM_Type:$lhs, LLVM_Type:$rhs, I32Attr:$lhs_rows,
                   I32Attr:$lhs_columns, I32Attr:$rhs_columns);
  string llvmBuilder = [{
    llvm::MatrixBuilder<decltype(builder)> mb(builder);
    $res = mb.CreateMatrixMultiply(
      $lhs, $rhs, $lhs_rows, $lhs_columns,
      $rhs_columns);
  }];
  let assemblyFormat = "$lhs `,` $rhs attr-dict "
    "`:` `(` type($lhs) `,` type($rhs) `)` `->` type($res)";
}

/// Create a llvm.matrix.transpose call, transposing a `rows` x `columns` 2-D
/// `matrix`, as specified in the LLVM MatrixBuilder.
def LLVM_MatrixTransposeOp : LLVM_OneResultOp<"intr.matrix.transpose"> {
  let arguments = (ins LLVM_Type:$matrix, I32Attr:$rows, I32Attr:$columns);
  string llvmBuilder = [{
    llvm::MatrixBuilder<decltype(builder)> mb(builder);
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
def LLVM_MaskedLoadOp : LLVM_OneResultOp<"intr.masked.load"> {
  let arguments = (ins LLVM_Type:$data, LLVM_Type:$mask,
                   Variadic<LLVM_Type>:$pass_thru, I32Attr:$alignment);
  string llvmBuilder = [{
    $res = $pass_thru.empty() ? builder.CreateMaskedLoad(
      $data, llvm::Align($alignment), $mask) :
      builder.CreateMaskedLoad(
        $data, llvm::Align($alignment), $mask, $pass_thru[0]);
  }];
  let assemblyFormat =
    "operands attr-dict `:` functional-type(operands, results)";
}

/// Create a call to Masked Store intrinsic.
def LLVM_MaskedStoreOp : LLVM_ZeroResultOp<"intr.masked.store"> {
  let arguments = (ins LLVM_Type:$value, LLVM_Type:$data, LLVM_Type:$mask,
                   I32Attr:$alignment);
  string llvmBuilder = [{
    builder.CreateMaskedStore(
      $value, $data, llvm::Align($alignment), $mask);
  }];
  let assemblyFormat = "$value `,` $data `,` $mask attr-dict `:` "
    "type($value) `,` type($mask) `into` type($data)";
}

/// Create a call to Masked Gather intrinsic.
def LLVM_masked_gather : LLVM_OneResultOp<"intr.masked.gather"> {
  let arguments = (ins LLVM_Type:$ptrs, LLVM_Type:$mask,
                   Variadic<LLVM_Type>:$pass_thru, I32Attr:$alignment);
  string llvmBuilder = [{
    $res = $pass_thru.empty() ? builder.CreateMaskedGather(
      $ptrs, llvm::Align($alignment), $mask) :
      builder.CreateMaskedGather(
        $ptrs, llvm::Align($alignment), $mask, $pass_thru[0]);
  }];
  let assemblyFormat =
    "operands attr-dict `:` functional-type(operands, results)";
}

/// Create a call to Masked Scatter intrinsic.
def LLVM_masked_scatter : LLVM_ZeroResultOp<"intr.masked.scatter"> {
  let arguments = (ins LLVM_Type:$value, LLVM_Type:$ptrs, LLVM_Type:$mask,
                   I32Attr:$alignment);
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

//
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

def LLVM_AtomicRMWType : AnyTypeOf<[LLVM_AnyFloat, LLVM_AnyInteger]>;

def LLVM_AtomicRMWOp : LLVM_Op<"atomicrmw">,
    Results<(outs LLVM_Type:$res)> {
  let arguments = (ins AtomicBinOp:$bin_op,
                   LLVM_PointerTo<LLVM_AtomicRMWType>:$ptr,
                   LLVM_AtomicRMWType:$val, AtomicOrdering:$ordering);
  let llvmBuilder = [{
    $res = builder.CreateAtomicRMW(getLLVMAtomicBinOp($bin_op), $ptr, $val,
                                   getLLVMAtomicOrdering($ordering));
  }];
  let parser = [{ return parseAtomicRMWOp(parser, result); }];
  let printer = [{ printAtomicRMWOp(p, *this); }];
  let verifier = "return ::verify(*this);";
}

def LLVM_AtomicCmpXchgType : AnyTypeOf<[LLVM_AnyInteger, LLVM_AnyPointer]>;

def LLVM_AtomicCmpXchgOp : LLVM_Op<"cmpxchg">, Results<(outs LLVM_Type:$res)> {
  let arguments = (ins LLVM_PointerTo<LLVM_AtomicCmpXchgType>:$ptr,
                   LLVM_AtomicCmpXchgType:$cmp, LLVM_AtomicCmpXchgType:$val,
                   AtomicOrdering:$success_ordering,
                   AtomicOrdering:$failure_ordering);
  let llvmBuilder = [{
    $res = builder.CreateAtomicCmpXchg($ptr, $cmp, $val,
                   getLLVMAtomicOrdering($success_ordering),
                   getLLVMAtomicOrdering($failure_ordering));
  }];
  let parser = [{ return parseAtomicCmpXchgOp(parser, result); }];
  let printer = [{ printAtomicCmpXchgOp(p, *this); }];
  let verifier = "return ::verify(*this);";
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

def LLVM_FenceOp : LLVM_ZeroResultOp<"fence", []> {
  let arguments = (ins AtomicOrdering:$ordering, StrAttr:$syncscope);
  let llvmBuilder = [{
    llvm::LLVMContext &llvmContext = builder.getContext();
    builder.CreateFence(getLLVMAtomicOrdering($ordering),
      llvmContext.getOrInsertSyncScopeID($syncscope));
  }];
  let parser = [{ return parseFenceOp(parser, result); }];
  let printer = [{ printFenceOp(p, *this); }];
  let verifier = "return ::verify(*this);";
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
          DefaultValuedAttr<AsmATTOrIntel, "AsmDialect::AD_ATT">>:$asm_dialect);

  let results = (outs Optional<LLVM_Type>:$res);

  let assemblyFormat = [{
    (`has_side_effects` $has_side_effects^)?
    (`is_align_stack` $is_align_stack^)?
    (`asm_dialect` `=` $asm_dialect^)?
    attr-dict
    $asm_string `,` $constraints
    operands `:` functional-type(operands, results)
   }];
}
#endif // LLVMIR_OPS
