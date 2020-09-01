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

def LLVM_OneResultOpBuilder : OpBuilder<
  "OpBuilder &, OperationState &result, Type resultType, "
  "ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
  [{
    if (resultType) result.addTypes(resultType);
    result.addOperands(operands);
    for (auto namedAttr : attributes) {
      result.addAttribute(namedAttr.first, namedAttr.second);
    }
  }]>;

def LLVM_ZeroResultOpBuilder : OpBuilder<
  "OpBuilder &, OperationState &result, ValueRange operands, "
  "ArrayRef<NamedAttribute> attributes = {}",
  [{
    result.addOperands(operands);
    for (auto namedAttr : attributes) {
      result.addAttribute(namedAttr.first, namedAttr.second);
    }
  }]>;

class LLVM_TwoBuilders<OpBuilder b1, OpBuilder b2> {
  list<OpBuilder> builders = [b1, b2];
}

// Base class for LLVM operations with one result.
class LLVM_OneResultOp<string mnemonic, list<OpTrait> traits = []> :
    LLVM_Op<mnemonic, traits>, Results<(outs LLVM_Type:$res)> {
  let builders = [LLVM_OneResultOpBuilder];
}

// Compatibility builder that takes an instance of wrapped llvm::VoidType
// to indicate no result.
def LLVM_VoidResultTypeOpBuilder : OpBuilder<
  "OpBuilder &builder, OperationState &result, Type resultType, "
  "ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
  [{
    auto llvmType = resultType.dyn_cast<LLVM::LLVMType>(); (void)llvmType;
    assert(llvmType && "result must be an LLVM type");
    assert(llvmType.isVoidTy() &&
           "for zero-result operands, only 'void' is accepted as result type");
    build(builder, result, operands, attributes);
  }]>;

// Base class for LLVM operations with zero results.
class LLVM_ZeroResultOp<string mnemonic, list<OpTrait> traits = []> :
    LLVM_Op<mnemonic, traits>, Results<(outs)>,
    LLVM_TwoBuilders<LLVM_VoidResultTypeOpBuilder, LLVM_ZeroResultOpBuilder>;

// Opaque builder used for terminator operations that contain successors.
def LLVM_TerminatorPassthroughOpBuilder : OpBuilder<
  "OpBuilder &, OperationState &result, ValueRange operands, "
  "SuccessorRange destinations, ArrayRef<NamedAttribute> attributes = {}",
  [{
    result.addOperands(operands);
    result.addSuccessors(destinations);
    result.addAttributes(attributes);
  }]>;

// Base class for LLVM terminator operations.  All terminator operations have
// zero results and an optional list of successors.
class LLVM_TerminatorOp<string mnemonic, list<OpTrait> traits = []> :
    LLVM_Op<mnemonic, !listconcat(traits, [Terminator])>;

// Class for arithmetic binary operations.
class LLVM_ArithmeticOp<string mnemonic, string builderFunc,
                        list<OpTrait> traits = []> :
    LLVM_OneResultOp<mnemonic,
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)>,
    Arguments<(ins LLVM_Type:$lhs, LLVM_Type:$rhs)>,
    LLVM_Builder<"$res = builder." # builderFunc # "($lhs, $rhs);"> {
  let parser = [{ return impl::parseOneResultSameOperandTypeOp(parser, result); }];
  let printer = [{ mlir::impl::printOneResultOp(this->getOperation(), p); }];
}
class LLVM_UnaryArithmeticOp<string mnemonic, string builderFunc,
                        list<OpTrait> traits = []> :
    LLVM_OneResultOp<mnemonic,
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)>,
    Arguments<(ins LLVM_Type:$operand)>,
    LLVM_Builder<"$res = builder." # builderFunc # "($operand);"> {
  let parser = [{ return impl::parseOneResultSameOperandTypeOp(parser, result); }];
  let printer = [{ mlir::impl::printOneResultOp(this->getOperation(), p); }];
}

// Integer binary operations.
def LLVM_AddOp : LLVM_ArithmeticOp<"add", "CreateAdd", [Commutative]>;
def LLVM_SubOp : LLVM_ArithmeticOp<"sub", "CreateSub">;
def LLVM_MulOp : LLVM_ArithmeticOp<"mul", "CreateMul", [Commutative]>;
def LLVM_UDivOp : LLVM_ArithmeticOp<"udiv", "CreateUDiv">;
def LLVM_SDivOp : LLVM_ArithmeticOp<"sdiv", "CreateSDiv">;
def LLVM_URemOp : LLVM_ArithmeticOp<"urem", "CreateURem">;
def LLVM_SRemOp : LLVM_ArithmeticOp<"srem", "CreateSRem">;
def LLVM_AndOp : LLVM_ArithmeticOp<"and", "CreateAnd">;
def LLVM_OrOp : LLVM_ArithmeticOp<"or", "CreateOr">;
def LLVM_XOrOp : LLVM_ArithmeticOp<"xor", "CreateXor">;
def LLVM_ShlOp : LLVM_ArithmeticOp<"shl", "CreateShl">;
def LLVM_LShrOp : LLVM_ArithmeticOp<"lshr", "CreateLShr">;
def LLVM_AShrOp : LLVM_ArithmeticOp<"ashr", "CreateAShr">;

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
def LLVM_ICmpOp : LLVM_OneResultOp<"icmp", [NoSideEffect]>,
                  Arguments<(ins ICmpPredicate:$predicate, LLVM_Type:$lhs,
                             LLVM_Type:$rhs)> {
  let llvmBuilder = [{
    $res = builder.CreateICmp(getLLVMCmpPredicate($predicate), $lhs, $rhs);
  }];
  let builders = [OpBuilder<
    "OpBuilder &b, OperationState &result, ICmpPredicate predicate, Value lhs, "
    "Value rhs", [{
      build(b, result, LLVMType::getInt1Ty(lhs.getType().getContext()),
            b.getI64IntegerAttr(static_cast<int64_t>(predicate)), lhs, rhs);
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
def LLVM_FCmpOp : LLVM_OneResultOp<"fcmp", [NoSideEffect]>,
                  Arguments<(ins FCmpPredicate:$predicate, LLVM_Type:$lhs,
                             LLVM_Type:$rhs)> {
  let llvmBuilder = [{
    $res = builder.CreateFCmp(getLLVMCmpPredicate($predicate), $lhs, $rhs);
  }];
  let builders = [OpBuilder<
    "OpBuilder &b, OperationState &result, FCmpPredicate predicate, Value lhs, "
    "Value rhs", [{
      build(b, result, LLVMType::getInt1Ty(lhs.getType().getContext()),
            b.getI64IntegerAttr(static_cast<int64_t>(predicate)), lhs, rhs);
    }]>];
  let parser = [{ return parseCmpOp<FCmpPredicate>(parser, result); }];
  let printer = [{ printFCmpOp(p, *this); }];
}

// Floating point binary operations.
def LLVM_FAddOp : LLVM_ArithmeticOp<"fadd", "CreateFAdd">;
def LLVM_FSubOp : LLVM_ArithmeticOp<"fsub", "CreateFSub">;
def LLVM_FMulOp : LLVM_ArithmeticOp<"fmul", "CreateFMul">;
def LLVM_FDivOp : LLVM_ArithmeticOp<"fdiv", "CreateFDiv">;
def LLVM_FRemOp : LLVM_ArithmeticOp<"frem", "CreateFRem">;
def LLVM_FNegOp : LLVM_UnaryArithmeticOp<"fneg", "CreateFNeg">;

// Common code definition that is used to verify and set the alignment attribute
// of LLVM ops that accept such an attribute.
class MemoryOpWithAlignmentBase {
  code alignmentVerifierCode = [{
    if (alignment().hasValue()) {
      auto align = alignment().getValue().getSExtValue();
      if (align < 0)
        return emitOpError("expected positive alignment");
    }
    return success();
  }];
  code setAlignmentCode = [{
    if ($alignment.hasValue()) {
      auto align = $alignment.getValue().getZExtValue();
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
    LLVM_OneResultOp<"alloca">,
    Arguments<(ins LLVM_Type:$arraySize, OptionalAttr<I64Attr>:$alignment)> {
  string llvmBuilder = [{
    auto *inst = builder.CreateAlloca(
      $_resultType->getPointerElementType(), $arraySize);
    }] # setAlignmentCode # [{
    $res = inst;
  }];
  let builders = [OpBuilder<
    "OpBuilder &b, OperationState &result, Type resultType, Value arraySize, "
    "unsigned alignment",
    [{
      if (alignment == 0)
        return build(b, result, resultType, arraySize, IntegerAttr());
      build(b, result, resultType, arraySize, b.getI64IntegerAttr(alignment));
  }]>];
  let parser = [{ return parseAllocaOp(parser, result); }];
  let printer = [{ printAllocaOp(p, *this); }];
  let verifier = alignmentVerifierCode;
}
def LLVM_GEPOp : LLVM_OneResultOp<"getelementptr", [NoSideEffect]>,
                 Arguments<(ins LLVM_Type:$base, Variadic<LLVM_Type>:$indices)>,
                 LLVM_Builder<"$res = builder.CreateGEP($base, $indices);"> {
  let assemblyFormat = [{
    $base `[` $indices `]` attr-dict `:` functional-type(operands, results)
  }];
}
def LLVM_LoadOp :
    MemoryOpWithAlignmentAndAttributes,
    LLVM_OneResultOp<"load">,
    Arguments<(ins LLVM_Type:$addr,
                   OptionalAttr<I64Attr>:$alignment,
                   UnitAttr:$volatile_,
                   UnitAttr:$nontemporal)> {
  string llvmBuilder = [{
    auto *inst = builder.CreateLoad($addr, $volatile_);
  }] # setAlignmentCode # setNonTemporalMetadataCode # [{
    $res = inst;
  }];
  let builders = [OpBuilder<
    "OpBuilder &b, OperationState &result, Value addr, "
    "unsigned alignment = 0, bool isVolatile = false, "
    "bool isNonTemporal = false",
    [{
      auto type = addr.getType().cast<LLVM::LLVMType>().getPointerElementTy();
      build(b, result, type, addr, alignment, isVolatile, isNonTemporal);
    }]>,
    OpBuilder<
    "OpBuilder &b, OperationState &result, Type t, Value addr, "
    "unsigned alignment = 0, bool isVolatile = false, "
    "bool isNonTemporal = false">];
  let parser = [{ return parseLoadOp(parser, result); }];
  let printer = [{ printLoadOp(p, *this); }];
  let verifier = alignmentVerifierCode;
}
def LLVM_StoreOp :
    MemoryOpWithAlignmentAndAttributes,
    LLVM_ZeroResultOp<"store">,
    Arguments<(ins LLVM_Type:$value,
                   LLVM_Type:$addr,
                   OptionalAttr<I64Attr>:$alignment,
                   UnitAttr:$volatile_,
                   UnitAttr:$nontemporal)> {
  string llvmBuilder = [{
    auto *inst = builder.CreateStore($value, $addr, $volatile_);
  }] # setAlignmentCode # setNonTemporalMetadataCode;
  let builders = [OpBuilder<
    "OpBuilder &b, OperationState &result, Value value, Value addr, "
    "unsigned alignment = 0, bool isVolatile = false, "
    "bool isNonTemporal = false">
    ];
  let parser = [{ return parseStoreOp(parser, result); }];
  let printer = [{ printStoreOp(p, *this); }];
  let verifier = alignmentVerifierCode;
}

// Casts.
class LLVM_CastOp<string mnemonic, string builderFunc,
                  list<OpTrait> traits = []> :
    LLVM_OneResultOp<mnemonic,
           !listconcat([NoSideEffect], traits)>,
    Arguments<(ins LLVM_Type:$arg)>,
    LLVM_Builder<"$res = builder." # builderFunc # "($arg, $_resultType);"> {
  let parser = [{ return mlir::impl::parseCastOp(parser, result); }];
  let printer = [{ mlir::impl::printCastOp(this->getOperation(), p); }];
}
def LLVM_BitcastOp : LLVM_CastOp<"bitcast", "CreateBitCast">;
def LLVM_AddrSpaceCastOp : LLVM_CastOp<"addrspacecast", "CreateAddrSpaceCast">;
def LLVM_IntToPtrOp : LLVM_CastOp<"inttoptr", "CreateIntToPtr">;
def LLVM_PtrToIntOp : LLVM_CastOp<"ptrtoint", "CreatePtrToInt">;
def LLVM_SExtOp : LLVM_CastOp<"sext", "CreateSExt">;
def LLVM_ZExtOp : LLVM_CastOp<"zext", "CreateZExt">;
def LLVM_TruncOp : LLVM_CastOp<"trunc", "CreateTrunc">;
def LLVM_SIToFPOp : LLVM_CastOp<"sitofp", "CreateSIToFP">;
def LLVM_UIToFPOp : LLVM_CastOp<"uitofp", "CreateUIToFP">;
def LLVM_FPToSIOp : LLVM_CastOp<"fptosi", "CreateFPToSI">;
def LLVM_FPToUIOp : LLVM_CastOp<"fptoui", "CreateFPToUI">;
def LLVM_FPExtOp : LLVM_CastOp<"fpext", "CreateFPExt">;
def LLVM_FPTruncOp : LLVM_CastOp<"fptrunc", "CreateFPTrunc">;

// Call-related operations.
def LLVM_InvokeOp : LLVM_Op<"invoke", [
                      AttrSizedOperandSegments,
                      DeclareOpInterfaceMethods<BranchOpInterface>,
                      Terminator
                    ]>,
                    Arguments<(ins OptionalAttr<FlatSymbolRefAttr>:$callee,
                               Variadic<LLVM_Type>:$operands,
                               Variadic<LLVM_Type>:$normalDestOperands,
                               Variadic<LLVM_Type>:$unwindDestOperands)>,
                    Results<(outs Variadic<LLVM_Type>)> {
  let successors = (successor AnySuccessor:$normalDest,
                              AnySuccessor:$unwindDest);

  let builders = [OpBuilder<
    "OpBuilder &b, OperationState &result, ArrayRef<Type> tys, "
    "FlatSymbolRefAttr callee, ValueRange ops, Block* normal, "
    "ValueRange normalOps, Block* unwind, ValueRange unwindOps",
    [{
      result.addAttribute("callee", callee);
      build(b, result, tys, ops, normal, normalOps, unwind, unwindOps);
    }]>,
    OpBuilder<
    "OpBuilder &b, OperationState &result, ArrayRef<Type> tys, "
    "ValueRange ops, Block* normal, "
    "ValueRange normalOps, Block* unwind, ValueRange unwindOps",
    [{
      build(b, result, tys, /*callee=*/FlatSymbolRefAttr(), ops, normalOps,
            unwindOps, normal, unwind);
    }]>];
  let verifier = [{ return ::verify(*this);  }];
  let parser = [{ return parseInvokeOp(parser, result); }];
  let printer = [{ printInvokeOp(p, *this); }];
}

def LLVM_LandingpadOp : LLVM_OneResultOp<"landingpad">,
                        Arguments<(ins UnitAttr:$cleanup,
                                       Variadic<LLVM_Type>)> {
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return parseLandingpadOp(parser, result); }];
  let printer = [{ printLandingpadOp(p, *this); }];
}

def LLVM_CallOp : LLVM_Op<"call">,
                  Arguments<(ins OptionalAttr<FlatSymbolRefAttr>:$callee,
                             Variadic<LLVM_Type>)>,
                  Results<(outs Variadic<LLVM_Type>)> {
  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, LLVMFuncOp func,"
    "ValueRange operands, ArrayRef<NamedAttribute> attributes = {}",
    [{
      LLVMType resultType = func.getType().getFunctionResultType();
      if (!resultType.isVoidTy())
        result.addTypes(resultType);
      result.addAttribute("callee", builder.getSymbolRefAttr(func));
      result.addAttributes(attributes);
      result.addOperands(operands);
    }]>];
  let verifier = [{
    if (getNumResults() > 1)
      return emitOpError("must have 0 or 1 result");
    return success();
  }];
  let parser = [{ return parseCallOp(parser, result); }];
  let printer = [{ printCallOp(p, *this); }];
}
def LLVM_ExtractElementOp : LLVM_OneResultOp<"extractelement", [NoSideEffect]>,
                          Arguments<(ins LLVM_Type:$vector,
                                     LLVM_Type:$position)> {
  string llvmBuilder = [{
    $res = builder.CreateExtractElement($vector, $position);
  }];
  let builders = [OpBuilder<
    "OpBuilder &b, OperationState &result, Value vector, Value position,"
    "ArrayRef<NamedAttribute> attrs = {}">];
  let parser = [{ return parseExtractElementOp(parser, result); }];
  let printer = [{ printExtractElementOp(p, *this); }];
}
def LLVM_ExtractValueOp : LLVM_OneResultOp<"extractvalue", [NoSideEffect]>,
                          Arguments<(ins LLVM_Type:$container,
                                     ArrayAttr:$position)> {
  string llvmBuilder = [{
    $res = builder.CreateExtractValue($container, extractPosition($position));
  }];
  let parser = [{ return parseExtractValueOp(parser, result); }];
  let printer = [{ printExtractValueOp(p, *this); }];
}
def LLVM_InsertElementOp : LLVM_OneResultOp<"insertelement", [NoSideEffect]>,
                         Arguments<(ins LLVM_Type:$vector, LLVM_Type:$value,
                                    LLVM_Type:$position)> {
  string llvmBuilder = [{
    $res = builder.CreateInsertElement($vector, $value, $position);
  }];
  let parser = [{ return parseInsertElementOp(parser, result); }];
  let printer = [{ printInsertElementOp(p, *this); }];
}
def LLVM_InsertValueOp : LLVM_OneResultOp<"insertvalue", [NoSideEffect]>,
                         Arguments<(ins LLVM_Type:$container, LLVM_Type:$value,
                                    ArrayAttr:$position)> {
  string llvmBuilder = [{
    $res = builder.CreateInsertValue($container, $value,
                                     extractPosition($position));
  }];
  let builders = [OpBuilder<
    "OpBuilder &b, OperationState &result, Value container, Value value, "
    "ArrayAttr position",
    [{
      build(b, result, container.getType(), container, value, position);
    }]>];
  let parser = [{ return parseInsertValueOp(parser, result); }];
  let printer = [{ printInsertValueOp(p, *this); }];
}
def LLVM_ShuffleVectorOp
    : LLVM_OneResultOp<"shufflevector", [NoSideEffect]>,
      Arguments<(ins LLVM_Type:$v1, LLVM_Type:$v2, ArrayAttr:$mask)> {
  string llvmBuilder = [{
      SmallVector<unsigned, 4> position = extractPosition($mask);
      SmallVector<int, 4> mask(position.begin(), position.end());
      $res = builder.CreateShuffleVector($v1, $v2, mask);
  }];
  let builders = [OpBuilder<
    "OpBuilder &b, OperationState &result, Value v1, Value v2, "
    "ArrayAttr mask, ArrayRef<NamedAttribute> attrs = {}">];
  let verifier = [{
    auto wrappedVectorType1 = v1().getType().cast<LLVM::LLVMType>();
    auto wrappedVectorType2 = v2().getType().cast<LLVM::LLVMType>();
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
      Arguments<(ins LLVM_Type:$condition, LLVM_Type:$trueValue,
                 LLVM_Type:$falseValue)>,
      LLVM_Builder<
          "$res = builder.CreateSelect($condition, $trueValue, $falseValue);"> {
  let builders = [OpBuilder<
    "OpBuilder &b, OperationState &result, Value condition, Value lhs, "
    "Value rhs", [{
      build(b, result, lhs.getType(), condition, lhs, rhs);
    }]>];
  let assemblyFormat = "operands attr-dict `:` type($condition) `,` type($res)";
}
def LLVM_FreezeOp : LLVM_OneResultOp<"freeze", [SameOperandsAndResultType]>,
                    Arguments<(ins LLVM_Type:$val)> {
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
  let arguments = (ins LLVMI1:$condition,
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

  let builders = [OpBuilder<
     "OpBuilder &builder, OperationState &result, Value condition,"
     "Block *trueDest, ValueRange trueOperands,"
     "Block *falseDest, ValueRange falseOperands,"
     "Optional<std::pair<uint32_t, uint32_t>> weights = {}", [{
        ElementsAttr weightsAttr;
        if (weights) {
          weightsAttr =
              builder.getI32VectorAttr({static_cast<int32_t>(weights->first),
                                       static_cast<int32_t>(weights->second)});
        }
        build(builder, result, condition, trueOperands, falseOperands, weightsAttr,
              trueDest, falseDest);
  }]>, OpBuilder<
    "OpBuilder &builder, OperationState &result, Value condition,"
    "Block *trueDest, Block *falseDest, ValueRange falseOperands = {}", [{
      build(builder, result, condition, trueDest, ValueRange(), falseDest,
            falseOperands);
  }]>, LLVM_TerminatorPassthroughOpBuilder];
}
def LLVM_ReturnOp : LLVM_TerminatorOp<"return", [NoSideEffect]>,
                    Arguments<(ins Variadic<LLVM_Type>:$args)> {
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


def LLVM_AddressOfOp
    : LLVM_OneResultOp<"mlir.addressof">,
      Arguments<(ins FlatSymbolRefAttr:$global_name)> {
  let summary = "Creates a pointer pointing to a global or a function";

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, LLVMType resType, "
              "StringRef name, ArrayRef<NamedAttribute> attrs = {}", [{
      result.addAttribute("global_name", builder.getSymbolRefAttr(name));
      result.addAttributes(attrs);
      result.addTypes(resType);}]>,

    OpBuilder<"OpBuilder &builder, OperationState &result, GlobalOp global, "
              "ArrayRef<NamedAttribute> attrs = {}", [{
      build(builder, result,
            global.getType().getPointerTo(global.addr_space().getZExtValue()),
            global.sym_name(), attrs);}]>,

    OpBuilder<"OpBuilder &builder, OperationState &result, LLVMFuncOp func, "
              "ArrayRef<NamedAttribute> attrs = {}", [{
      build(builder, result,
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
    OpBuilder<"OpBuilder &builder, OperationState &result, LLVMType type, "
              "bool isConstant, Linkage linkage, StringRef name, "
              "Attribute value, unsigned addrSpace = 0, "
              "ArrayRef<NamedAttribute> attrs = {}">
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
                                 FunctionLike, Symbol]>,
      Arguments<(ins DefaultValuedAttr<Linkage,
                                       "Linkage::External">:$linkage,
                     OptionalAttr<FlatSymbolRefAttr>:$personality,
                     OptionalAttr<ArrayAttr>:$passthrough)> {
  let summary = "LLVM dialect function, has wrapped LLVM IR function type";

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, StringRef name, "
              "LLVMType type, LLVM::Linkage linkage = LLVM::Linkage::External, "
              "ArrayRef<NamedAttribute> attrs = {}, "
              "ArrayRef<MutableDictionaryAttr> argAttrs = {}">
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
      Arguments<(ins AnyAttr:$value)>,
      LLVM_Builder<"$res = getLLVMConstant($_resultType, $value, $_location);">
{
  let assemblyFormat = "`(` $value `)` attr-dict `:` type($res)";
  let verifier = [{ return ::verify(*this); }];
}

def LLVM_DialectCastOp : LLVM_Op<"mlir.cast", [NoSideEffect]>,
                         Results<(outs AnyType:$res)>,
                         Arguments<(ins AnyType:$in)> {
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
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($res)";
  let verifier = "return ::verify(*this);";
}

// Operations that correspond to LLVM intrinsics. With MLIR operation set being
// extendable, there is no reason to introduce a hard boundary between "core"
// operations and intrinsics. However, we systematically prefix them with
// "intr." to avoid potential name clashes.

class LLVM_UnaryIntrinsicOp<string func, list<OpTrait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)>,
    Arguments<(ins LLVM_Type:$in)>;

class LLVM_BinarySameArgsIntrinsicOp<string func, list<OpTrait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)>,
    Arguments<(ins LLVM_Type:$a, LLVM_Type:$b)>;

class LLVM_TernarySameArgsIntrinsicOp<string func, list<OpTrait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([NoSideEffect, SameOperandsAndResultType], traits)>,
    Arguments<(ins LLVM_Type:$a, LLVM_Type:$b, LLVM_Type:$c)>;

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
def LLVM_Prefetch : LLVM_ZeroResultIntrOp<"prefetch", [0]>,
                    Arguments<(ins LLVM_Type:$addr, LLVM_Type:$rw,
                               LLVM_Type:$hint, LLVM_Type:$cache)>;
def LLVM_SinOp : LLVM_UnaryIntrinsicOp<"sin">;
def LLVM_SqrtOp : LLVM_UnaryIntrinsicOp<"sqrt">;
def LLVM_PowOp : LLVM_BinarySameArgsIntrinsicOp<"pow">;
def LLVM_BitReverseOp : LLVM_UnaryIntrinsicOp<"bitreverse">;
def LLVM_CtPopOp : LLVM_UnaryIntrinsicOp<"ctpop">;
def LLVM_MaxNumOp : LLVM_BinarySameArgsIntrinsicOp<"maxnum">;
def LLVM_MinNumOp : LLVM_BinarySameArgsIntrinsicOp<"minnum">;
def LLVM_SMaxOp : LLVM_BinarySameArgsIntrinsicOp<"smax">;
def LLVM_SMinOp : LLVM_BinarySameArgsIntrinsicOp<"smin">;

def LLVM_MemcpyOp : LLVM_ZeroResultIntrOp<"memcpy", [0, 1, 2]>,
                    Arguments<(ins LLVM_Type:$dst, LLVM_Type:$src,
                               LLVM_Type:$len, LLVM_Type:$isVolatile)>;
def LLVM_MemcpyInlineOp : LLVM_ZeroResultIntrOp<"memcpy.inline", [0, 1, 2]>,
                          Arguments<(ins LLVM_Type:$dst, LLVM_Type:$src,
                                     LLVM_Type:$len, LLVM_Type:$isVolatile)>;

//
// Vector Reductions.
//

def LLVM_experimental_vector_reduce_add : LLVM_VectorReduction<"add">;
def LLVM_experimental_vector_reduce_and : LLVM_VectorReduction<"and">;
def LLVM_experimental_vector_reduce_mul : LLVM_VectorReduction<"mul">;
def LLVM_experimental_vector_reduce_fmax : LLVM_VectorReduction<"fmax">;
def LLVM_experimental_vector_reduce_fmin : LLVM_VectorReduction<"fmin">;
def LLVM_experimental_vector_reduce_or : LLVM_VectorReduction<"or">;
def LLVM_experimental_vector_reduce_smax : LLVM_VectorReduction<"smax">;
def LLVM_experimental_vector_reduce_smin : LLVM_VectorReduction<"smin">;
def LLVM_experimental_vector_reduce_umax : LLVM_VectorReduction<"umax">;
def LLVM_experimental_vector_reduce_umin : LLVM_VectorReduction<"umin">;
def LLVM_experimental_vector_reduce_xor : LLVM_VectorReduction<"xor">;

def LLVM_experimental_vector_reduce_v2_fadd : LLVM_VectorReductionV2<"fadd">;
def LLVM_experimental_vector_reduce_v2_fmul : LLVM_VectorReductionV2<"fmul">;

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
    : LLVM_OneResultOp<"intr.matrix.column.major.load">,
      Arguments<(ins LLVM_Type:$data, LLVM_Type:$stride, I1Attr:$isVolatile,
                 I32Attr:$rows, I32Attr:$columns)> {
  string llvmBuilder = [{
    llvm::MatrixBuilder<decltype(builder)> mb(builder);
    const llvm::DataLayout &dl =
      builder.GetInsertBlock()->getModule()->getDataLayout();
    llvm::Align align = dl.getABITypeAlign(
      $data->getType()->getPointerElementType());
    $res = mb.CreateColumnMajorLoad(
      $data, align, $stride, $isVolatile.getZExtValue(), $rows.getZExtValue(),
      $columns.getZExtValue());
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
    : LLVM_ZeroResultOp<"intr.matrix.column.major.store">,
      Arguments<(ins LLVM_Type:$matrix, LLVM_Type:$data, LLVM_Type:$stride,
                 I1Attr:$isVolatile, I32Attr:$rows, I32Attr:$columns)> {
  string llvmBuilder = [{
    llvm::MatrixBuilder<decltype(builder)> mb(builder);
    const llvm::DataLayout &dl =
      builder.GetInsertBlock()->getModule()->getDataLayout();
    llvm::Align align = dl.getABITypeAlign(
      $data->getType()->getPointerElementType());
    mb.CreateColumnMajorStore(
      $matrix, $data, align, $stride, $isVolatile.getZExtValue(),
      $rows.getZExtValue(), $columns.getZExtValue());
  }];
  let assemblyFormat = "$matrix `,` $data `,` `<` `stride` `=` $stride `>` "
    "attr-dict`:` type($matrix) `to` type($data) `stride` type($stride)";
}

/// Create a llvm.matrix.multiply call, multiplying 2-D matrices LHS and RHS, as
/// specified in the LLVM MatrixBuilder.
def LLVM_MatrixMultiplyOp
    : LLVM_OneResultOp<"intr.matrix.multiply">,
      Arguments<(
        ins LLVM_Type:$lhs, LLVM_Type:$rhs,
            I32Attr:$lhs_rows, I32Attr:$lhs_columns, I32Attr:$rhs_columns)> {
  string llvmBuilder = [{
    llvm::MatrixBuilder<decltype(builder)> mb(builder);
    $res = mb.CreateMatrixMultiply(
      $lhs, $rhs, $lhs_rows.getZExtValue(), $lhs_columns.getZExtValue(),
      $rhs_columns.getZExtValue());
  }];
  let assemblyFormat = "$lhs `,` $rhs attr-dict "
    "`:` `(` type($lhs) `,` type($rhs) `)` `->` type($res)";
}

/// Create a llvm.matrix.transpose call, transposing a `rows` x `columns` 2-D
/// `matrix`, as specified in the LLVM MatrixBuilder.
def LLVM_MatrixTransposeOp
    : LLVM_OneResultOp<"intr.matrix.transpose">,
      Arguments<(ins LLVM_Type:$matrix, I32Attr:$rows, I32Attr:$columns)> {
  string llvmBuilder = [{
    llvm::MatrixBuilder<decltype(builder)> mb(builder);
    $res = mb.CreateMatrixTranspose(
      $matrix, $rows.getZExtValue(), $columns.getZExtValue());
  }];
  let assemblyFormat = "$matrix attr-dict `:` type($matrix) `into` type($res)";
}

//
// LLVM masked operations.
//

/// Create a llvm.get.active.lane.mask to set a mask up to a given position.
def LLVM_GetActiveLaneMaskOp
    : LLVM_OneResultIntrOp<"get.active.lane.mask", [0], [0], [NoSideEffect]>,
      Arguments<(ins LLVM_Type:$base, LLVM_Type:$n)> {
  let assemblyFormat = "$base `,` $n attr-dict `:` "
    "type($base) `,` type($n) `to` type($res)";
}

/// Create a call to Masked Load intrinsic.
def LLVM_MaskedLoadOp
    : LLVM_OneResultOp<"intr.masked.load">,
      Arguments<(ins LLVM_Type:$data, LLVM_Type:$mask,
                 Variadic<LLVM_Type>:$pass_thru, I32Attr:$alignment)> {
  string llvmBuilder = [{
    $res = $pass_thru.empty() ? builder.CreateMaskedLoad(
      $data, llvm::Align($alignment.getZExtValue()), $mask) :
      builder.CreateMaskedLoad(
        $data, llvm::Align($alignment.getZExtValue()), $mask, $pass_thru[0]);
  }];
  let assemblyFormat =
    "operands attr-dict `:` functional-type(operands, results)";
}

/// Create a call to Masked Store intrinsic.
def LLVM_MaskedStoreOp
    : LLVM_ZeroResultOp<"intr.masked.store">,
      Arguments<(ins LLVM_Type:$value, LLVM_Type:$data, LLVM_Type:$mask,
                 I32Attr:$alignment)> {
  string llvmBuilder = [{
    builder.CreateMaskedStore(
      $value, $data, llvm::Align($alignment.getZExtValue()), $mask);
  }];
  let assemblyFormat = "$value `,` $data `,` $mask attr-dict `:` "
    "type($value) `,` type($mask) `into` type($data)";
}

/// Create a call to Masked Gather intrinsic.
def LLVM_masked_gather
    : LLVM_OneResultOp<"intr.masked.gather">,
      Arguments<(ins LLVM_Type:$ptrs, LLVM_Type:$mask,
                 Variadic<LLVM_Type>:$pass_thru, I32Attr:$alignment)> {
  string llvmBuilder = [{
    $res = $pass_thru.empty() ? builder.CreateMaskedGather(
      $ptrs, llvm::Align($alignment.getZExtValue()), $mask) :
      builder.CreateMaskedGather(
        $ptrs, llvm::Align($alignment.getZExtValue()), $mask, $pass_thru[0]);
  }];
  let assemblyFormat =
    "operands attr-dict `:` functional-type(operands, results)";
}

/// Create a call to Masked Scatter intrinsic.
def LLVM_masked_scatter
    : LLVM_ZeroResultOp<"intr.masked.scatter">,
      Arguments<(ins LLVM_Type:$value, LLVM_Type:$ptrs, LLVM_Type:$mask,
                 I32Attr:$alignment)> {
  string llvmBuilder = [{
    builder.CreateMaskedScatter(
      $value, $ptrs, llvm::Align($alignment.getZExtValue()), $mask);
  }];
  let assemblyFormat = "$value `,` $ptrs `,` $mask attr-dict `:` "
    "type($value) `,` type($mask) `into` type($ptrs)";
}

/// Create a call to Masked Expand Load intrinsic.
def LLVM_masked_expandload
    : LLVM_IntrOp<"masked.expandload", [0], [], [], 1>,
      Arguments<(ins LLVM_Type, LLVM_Type, LLVM_Type)>;

/// Create a call to Masked Compress Store intrinsic.
def LLVM_masked_compressstore
    : LLVM_IntrOp<"masked.compressstore", [], [0], [], 0>,
    Arguments<(ins LLVM_Type, LLVM_Type, LLVM_Type)>;

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

def LLVM_AtomicRMWOp : LLVM_Op<"atomicrmw">,
    Arguments<(ins AtomicBinOp:$bin_op, LLVM_Type:$ptr, LLVM_Type:$val,
                   AtomicOrdering:$ordering)>,
    Results<(outs LLVM_Type:$res)> {
  let llvmBuilder = [{
    $res = builder.CreateAtomicRMW(getLLVMAtomicBinOp($bin_op), $ptr, $val,
                                   getLLVMAtomicOrdering($ordering));
  }];
  let parser = [{ return parseAtomicRMWOp(parser, result); }];
  let printer = [{ printAtomicRMWOp(p, *this); }];
  let verifier = "return ::verify(*this);";
}

def LLVM_AtomicCmpXchgOp : LLVM_Op<"cmpxchg">,
    Arguments<(ins LLVM_Type:$ptr, LLVM_Type:$cmp, LLVM_Type:$val,
                   AtomicOrdering:$success_ordering,
                   AtomicOrdering:$failure_ordering)>,
    Results<(outs LLVM_Type:$res)> {
  let llvmBuilder = [{
    $res = builder.CreateAtomicCmpXchg($ptr, $cmp, $val,
                   getLLVMAtomicOrdering($success_ordering),
                   getLLVMAtomicOrdering($failure_ordering));
  }];
  let parser = [{ return parseAtomicCmpXchgOp(parser, result); }];
  let printer = [{ printAtomicCmpXchgOp(p, *this); }];
  let verifier = "return ::verify(*this);";
}

def LLVM_AssumeOp : LLVM_Op<"intr.assume", []>,
                    Arguments<(ins LLVM_Type:$cond)> {
  let llvmBuilder = [{
    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn =
        llvm::Intrinsic::getDeclaration(module, llvm::Intrinsic::assume, {});
    builder.CreateCall(fn, {$cond});
  }];
}

def LLVM_FenceOp : LLVM_ZeroResultOp<"fence", []>,
                   Arguments<(ins AtomicOrdering:$ordering,
                    StrAttr:$syncscope)> {
  let llvmBuilder = [{
    llvm::LLVMContext &llvmContext = builder.getContext();
    builder.CreateFence(getLLVMAtomicOrdering($ordering),
      llvmContext.getOrInsertSyncScopeID($syncscope));
  }];
  let parser = [{ return parseFenceOp(parser, result); }];
  let printer = [{ printFenceOp(p, *this); }];
  let verifier = "return ::verify(*this);";
}
#endif // LLVMIR_OPS
