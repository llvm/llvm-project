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

include "mlir/Dialect/LLVMIR/LLVMAttrDefs.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class LLVM_Builder<string builder> {
  string llvmBuilder = builder;
}

// Base class for LLVM terminator operations.  All terminator operations have
// zero results and an optional list of successors.
class LLVM_TerminatorOp<string mnemonic, list<Trait> traits = []> :
    LLVM_Op<mnemonic, !listconcat(traits, [Terminator])>;

// Class for arithmetic binary operations.
class LLVM_ArithmeticOpBase<Type type, string mnemonic,
                            string instName, list<Trait> traits = []> :
    LLVM_Op<mnemonic,
           !listconcat([Pure, SameOperandsAndResultType], traits)>,
    LLVM_Builder<"$res = builder.Create" # instName # "($lhs, $rhs);"> {
  dag commonArgs = (ins LLVM_ScalarOrVectorOf<type>:$lhs,
                    LLVM_ScalarOrVectorOf<type>:$rhs);
  let results = (outs LLVM_ScalarOrVectorOf<type>:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "$lhs `,` $rhs custom<LLVMOpAttrs>(attr-dict) `:` type($res)";
  string llvmInstName = instName;
  string mlirBuilder = [{
    $res = $_builder.create<$_qualCppClassName>($_location, $lhs, $rhs);
  }];
}
class LLVM_IntArithmeticOp<string mnemonic, string instName,
                           list<Trait> traits = []> :
    LLVM_ArithmeticOpBase<AnyInteger, mnemonic, instName, traits> {
  let arguments = commonArgs;
}
class LLVM_FloatArithmeticOp<string mnemonic, string instName,
                             list<Trait> traits = []> :
    LLVM_ArithmeticOpBase<LLVM_AnyFloat, mnemonic, instName,
    !listconcat([DeclareOpInterfaceMethods<FastmathFlagsInterface>], traits)> {
  dag fmfArg = (
    ins DefaultValuedAttr<LLVM_FastmathFlagsAttr, "{}">:$fastmathFlags);
  let arguments = !con(commonArgs, fmfArg);
}

// Class for arithmetic unary operations.
class LLVM_UnaryFloatArithmeticOp<Type type, string mnemonic,
                                  string instName, list<Trait> traits = []> :
    LLVM_Op<mnemonic,
           !listconcat([Pure, SameOperandsAndResultType, DeclareOpInterfaceMethods<FastmathFlagsInterface>], traits)>,
    LLVM_Builder<"$res = builder.Create" # instName # "($operand);"> {
  let arguments = (
    ins type:$operand,
    DefaultValuedAttr<LLVM_FastmathFlagsAttr, "{}">:$fastmathFlags);
  let results = (outs type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "$operand custom<LLVMOpAttrs>(attr-dict) `:` type($res)";
  string llvmInstName = instName;
  string mlirBuilder = [{
    $res = $_builder.create<$_qualCppClassName>($_location, $operand);
  }];
}

// Integer binary operations.
def LLVM_AddOp : LLVM_IntArithmeticOp<"add", "Add", [Commutative]>;
def LLVM_SubOp : LLVM_IntArithmeticOp<"sub", "Sub">;
def LLVM_MulOp : LLVM_IntArithmeticOp<"mul", "Mul", [Commutative]>;
def LLVM_UDivOp : LLVM_IntArithmeticOp<"udiv", "UDiv">;
def LLVM_SDivOp : LLVM_IntArithmeticOp<"sdiv", "SDiv">;
def LLVM_URemOp : LLVM_IntArithmeticOp<"urem", "URem">;
def LLVM_SRemOp : LLVM_IntArithmeticOp<"srem", "SRem">;
def LLVM_AndOp : LLVM_IntArithmeticOp<"and", "And">;
def LLVM_OrOp : LLVM_IntArithmeticOp<"or", "Or">;
def LLVM_XOrOp : LLVM_IntArithmeticOp<"xor", "Xor">;
def LLVM_ShlOp : LLVM_IntArithmeticOp<"shl", "Shl">;
def LLVM_LShrOp : LLVM_IntArithmeticOp<"lshr", "LShr">;
def LLVM_AShrOp : LLVM_IntArithmeticOp<"ashr", "AShr">;

// Base class for compare operations. A compare operation takes two operands
// of the same type and returns a boolean result. If the operands are
// vectors, then the result has to be a boolean vector of the same shape.
class LLVM_ArithmeticCmpOp<string mnemonic, list<Trait> traits = []> :
    LLVM_Op<mnemonic, traits # [SameTypeOperands, TypesMatchWith<
    "result type has i1 element type and same shape as operands",
    "lhs", "res", "::getI1SameShape($_self)">]> {
  let results = (outs LLVM_ScalarOrVectorOf<I1>:$res);
}

// Other integer operations.
def LLVM_ICmpOp : LLVM_ArithmeticCmpOp<"icmp", [Pure]> {
  let arguments = (ins ICmpPredicate:$predicate,
                   AnyTypeOf<[LLVM_ScalarOrVectorOf<AnyInteger>,
                              LLVM_ScalarOrVectorOf<LLVM_AnyPointer>]>:$lhs,
                   AnyTypeOf<[LLVM_ScalarOrVectorOf<AnyInteger>,
                              LLVM_ScalarOrVectorOf<LLVM_AnyPointer>]>:$rhs);
  let builders = [
    OpBuilder<(ins "ICmpPredicate":$predicate, "Value":$lhs, "Value":$rhs)>
  ];
  let hasCustomAssemblyFormat = 1;
  string llvmInstName = "ICmp";
  string llvmBuilder = [{
    $res = builder.CreateICmp(getLLVMCmpPredicate($predicate), $lhs, $rhs);
  }];
  string mlirBuilder = [{
    auto *iCmpInst = cast<llvm::ICmpInst>(inst);
    $res = $_builder.create<$_qualCppClassName>(
      $_location, getICmpPredicate(iCmpInst->getPredicate()), $lhs, $rhs);
  }];
  // Set the $predicate index to -1 to indicate there is no matching operand
  // and decrement the following indices.
  list<int> llvmArgIndices = [-1, 0, 1];
}

// Other floating-point operations.
def LLVM_FCmpOp : LLVM_ArithmeticCmpOp<"fcmp", [
    Pure, DeclareOpInterfaceMethods<FastmathFlagsInterface>]> {
  let arguments = (ins FCmpPredicate:$predicate,
                   LLVM_ScalarOrVectorOf<LLVM_AnyFloat>:$lhs,
                   LLVM_ScalarOrVectorOf<LLVM_AnyFloat>:$rhs,
                   DefaultValuedAttr<LLVM_FastmathFlagsAttr,
                                     "{}">:$fastmathFlags);
  let builders = [
    OpBuilder<(ins "FCmpPredicate":$predicate, "Value":$lhs, "Value":$rhs)>
  ];
  let hasCustomAssemblyFormat = 1;
  string llvmInstName = "FCmp";
  string llvmBuilder = [{
    $res = builder.CreateFCmp(getLLVMCmpPredicate($predicate), $lhs, $rhs);
  }];
  // FIXME: Import fastmath flags.
  string mlirBuilder = [{
    auto *fCmpInst = cast<llvm::FCmpInst>(inst);
    $res = $_builder.create<$_qualCppClassName>(
      $_location, getFCmpPredicate(fCmpInst->getPredicate()), $lhs, $rhs);
  }];
  // Set the $predicate index to -1 to indicate there is no matching operand
  // and decrement the following indices.
  list<int> llvmArgIndices = [-1, 0, 1, 2];
}

// Floating point binary operations.
def LLVM_FAddOp : LLVM_FloatArithmeticOp<"fadd", "FAdd">;
def LLVM_FSubOp : LLVM_FloatArithmeticOp<"fsub", "FSub">;
def LLVM_FMulOp : LLVM_FloatArithmeticOp<"fmul", "FMul">;
def LLVM_FDivOp : LLVM_FloatArithmeticOp<"fdiv", "FDiv">;
def LLVM_FRemOp : LLVM_FloatArithmeticOp<"frem", "FRem">;
def LLVM_FNegOp : LLVM_UnaryFloatArithmeticOp<
  LLVM_ScalarOrVectorOf<LLVM_AnyFloat>, "fneg", "FNeg">;

// Common code definition that is used to verify and set the alignment attribute
// of LLVM ops that accept such an attribute.
class MemoryOpWithAlignmentBase {
  code setAlignmentCode = [{
    if ($alignment.has_value()) {
      auto align = $alignment.value();
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
                   OptionalAttr<I64Attr>:$alignment,
                   OptionalAttr<TypeAttr>:$elem_type);
  let results = (outs Res<LLVM_AnyPointer, "",
                          [MemAlloc<AutomaticAllocationScopeResource>]>:$res);
  string llvmInstName = "Alloca";
  string llvmBuilder = [{
    auto addrSpace = $_resultType->getPointerAddressSpace();
    llvm::Type *elementType = moduleTranslation.convertType(
        $elem_type ? *$elem_type
                   : op.getType().cast<LLVMPointerType>().getElementType());
    auto *inst = builder.CreateAlloca(elementType, addrSpace, $arraySize);
    }] # setAlignmentCode # [{
    $res = inst;
  }];
  // FIXME: Import attributes.
  string mlirBuilder = [{
    auto *allocaInst = cast<llvm::AllocaInst>(inst);
    Type allocatedType = convertType(allocaInst->getAllocatedType());
    unsigned alignment = allocaInst->getAlign().value();
    $res = $_builder.create<LLVM::AllocaOp>(
      $_location, $_resultType, allocatedType, $arraySize, alignment);
  }];
  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$arraySize,
               "unsigned":$alignment),
    [{
      assert(!resultType.cast<LLVMPointerType>().isOpaque() &&
             "pass the allocated type explicitly if opaque pointers are used");
      if (alignment == 0)
        return build($_builder, $_state, resultType, arraySize, IntegerAttr(),
                     TypeAttr());
      build($_builder, $_state, resultType, arraySize,
        $_builder.getI64IntegerAttr(alignment), TypeAttr());
    }]>,
    OpBuilder<(ins "Type":$resultType, "Type":$elementType, "Value":$arraySize,
               CArg<"unsigned", "0">:$alignment),
    [{
      TypeAttr elemTypeAttr =
          resultType.cast<LLVMPointerType>().isOpaque() ?
          TypeAttr::get(elementType) : TypeAttr();
      build($_builder, $_state, resultType, arraySize,
            alignment == 0 ? IntegerAttr()
                           : $_builder.getI64IntegerAttr(alignment),
            elemTypeAttr);

    }]>
    ];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LLVM_GEPOp : LLVM_Op<"getelementptr", [Pure]> {
  let arguments = (ins LLVM_ScalarOrVectorOf<LLVM_AnyPointer>:$base,
                   Variadic<LLVM_ScalarOrVectorOf<AnyInteger>>:$dynamicIndices,
                   DenseI32ArrayAttr:$rawConstantIndices,
                   OptionalAttr<TypeAttr>:$elem_type);
  let results = (outs LLVM_ScalarOrVectorOf<LLVM_AnyPointer>:$res);
  let skipDefaultBuilders = 1;

  let description = [{
    This operation mirrors LLVM IRs 'getelementptr' operation that is used to
    perform pointer arithmetic.

    Like in LLVM IR, it is possible to use both constants as well as SSA values
    as indices. In the case of indexing within a structure, it is required to
    either use constant indices directly, or supply a constant SSA value.

    Examples:

    ```mlir
    // GEP with an SSA value offset
    %0 = llvm.getelementptr %1[%2] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>

    // GEP with a constant offset
    %0 = llvm.getelementptr %1[3] : (!llvm.ptr<f32>) -> !llvm.ptr<f32>

    // GEP with constant offsets into a structure
    %0 = llvm.getelementptr %1[0, 1]
       : (!llvm.ptr<struct(i32, f32)>) -> !llvm.ptr<f32>
    ```
  }];

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Type":$basePtrType, "Value":$basePtr,
               "ValueRange":$indices,
               CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "Type":$resultType, "Value":$basePtr, "ValueRange":$indices,
               CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "Type":$resultType, "Value":$basePtr, "ArrayRef<GEPArg>":$indices,
               CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "Type":$resultType, "Type":$basePtrType, "Value":$basePtr,
               "ArrayRef<GEPArg>":$indices,
               CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];
  let llvmBuilder = [{
    SmallVector<llvm::Value *> indices;
    indices.reserve($rawConstantIndices.size());
    GEPIndicesAdaptor<decltype($dynamicIndices)>
        gepIndices(op.getRawConstantIndicesAttr(), $dynamicIndices);
    for (PointerUnion<IntegerAttr, llvm::Value*> valueOrAttr : gepIndices) {
      if (llvm::Value* value = valueOrAttr.dyn_cast<llvm::Value*>())
        indices.push_back(value);
      else
        indices.push_back(
            builder.getInt32(valueOrAttr.get<IntegerAttr>().getInt()));
    }
    Type baseElementType = op.getSourceElementType();
    llvm::Type *elementType = moduleTranslation.convertType(baseElementType);
    $res = builder.CreateGEP(elementType, $base, indices);
  }];
  let assemblyFormat = [{
    $base `[` custom<GEPIndices>($dynamicIndices, $rawConstantIndices) `]` attr-dict
    `:` functional-type(operands, results) (`,` $elem_type^)?
  }];

  let extraClassDeclaration = [{
    constexpr static int32_t kDynamicIndex = std::numeric_limits<int32_t>::min();

    /// Returns the type pointed to by the pointer argument of this GEP.
    Type getSourceElementType();

    GEPIndicesAdaptor<ValueRange> getIndices();
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
  string llvmInstName = "Load";
  string llvmBuilder = [{
    auto *inst = builder.CreateLoad($_resultType, $addr, $volatile_);
  }] # setAlignmentCode
     # setNonTemporalMetadataCode
     # setAccessGroupsMetadataCode
     # setAliasScopeMetadataCode
     # [{
    $res = inst;
  }];
  // FIXME: Import attributes.
  string mlirBuilder = [{
    $res = $_builder.create<LLVM::LoadOp>($_location, $_resultType, $addr);
  }];
  let builders = [
    OpBuilder<(ins "Value":$addr, CArg<"unsigned", "0">:$alignment,
      CArg<"bool", "false">:$isVolatile, CArg<"bool", "false">:$isNonTemporal),
    [{
      auto type = addr.getType().cast<LLVMPointerType>().getElementType();
      assert(type && "must provide explicit element type to the constructor "
                     "when the pointer type is opaque");
      build($_builder, $_state, type, addr, alignment, isVolatile, isNonTemporal);
    }]>,
    OpBuilder<(ins "Type":$t, "Value":$addr,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isVolatile,
      CArg<"bool", "false">:$isNonTemporal)>,];
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
  string llvmInstName = "Store";
  string llvmBuilder = [{
    auto *inst = builder.CreateStore($value, $addr, $volatile_);
  }] # setAlignmentCode
     # setNonTemporalMetadataCode
     # setAccessGroupsMetadataCode
     # setAliasScopeMetadataCode;
  // FIXME: Import attributes.
  string mlirBuilder = [{
    $_builder.create<LLVM::StoreOp>($_location, $value, $addr);
  }];
  let builders = [
    OpBuilder<(ins "Value":$value, "Value":$addr,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isVolatile,
      CArg<"bool", "false">:$isNonTemporal)>
    ];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// Casts.
class LLVM_CastOp<string mnemonic, string instName, Type type,
                  Type resultType, list<Trait> traits = []> :
    LLVM_Op<mnemonic, !listconcat([Pure], traits)>,
    LLVM_Builder<"$res = builder.Create" # instName # "($arg, $_resultType);"> {
  let arguments = (ins type:$arg);
  let results = (outs resultType:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
  string llvmInstName = instName;
  string mlirBuilder = [{
    $res = $_builder.create<$_qualCppClassName>(
      $_location, $_resultType, $arg);
  }];
}
def LLVM_BitcastOp : LLVM_CastOp<"bitcast", "BitCast",
                                 LLVM_AnyNonAggregate, LLVM_AnyNonAggregate> {
  let hasFolder = 1;
}
def LLVM_AddrSpaceCastOp : LLVM_CastOp<"addrspacecast", "AddrSpaceCast",
                                       LLVM_ScalarOrVectorOf<LLVM_AnyPointer>,
                                       LLVM_ScalarOrVectorOf<LLVM_AnyPointer>> {
  let hasFolder = 1;
}
def LLVM_IntToPtrOp : LLVM_CastOp<"inttoptr", "IntToPtr",
                                  LLVM_ScalarOrVectorOf<AnyInteger>,
                                  LLVM_ScalarOrVectorOf<LLVM_AnyPointer>>;
def LLVM_PtrToIntOp : LLVM_CastOp<"ptrtoint", "PtrToInt",
                                  LLVM_ScalarOrVectorOf<LLVM_AnyPointer>,
                                  LLVM_ScalarOrVectorOf<AnyInteger>>;
def LLVM_SExtOp : LLVM_CastOp<"sext", "SExt",
                              LLVM_ScalarOrVectorOf<AnyInteger>,
                              LLVM_ScalarOrVectorOf<AnyInteger>>;
def LLVM_ZExtOp : LLVM_CastOp<"zext", "ZExt",
                              LLVM_ScalarOrVectorOf<AnyInteger>,
                              LLVM_ScalarOrVectorOf<AnyInteger>>;
def LLVM_TruncOp : LLVM_CastOp<"trunc", "Trunc",
                               LLVM_ScalarOrVectorOf<AnyInteger>,
                               LLVM_ScalarOrVectorOf<AnyInteger>>;
def LLVM_SIToFPOp : LLVM_CastOp<"sitofp", "SIToFP",
                                LLVM_ScalarOrVectorOf<AnyInteger>,
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>>;
def LLVM_UIToFPOp : LLVM_CastOp<"uitofp", "UIToFP",
                                LLVM_ScalarOrVectorOf<AnyInteger>,
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>>;
def LLVM_FPToSIOp : LLVM_CastOp<"fptosi", "FPToSI",
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>,
                                LLVM_ScalarOrVectorOf<AnyInteger>>;
def LLVM_FPToUIOp : LLVM_CastOp<"fptoui", "FPToUI",
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>,
                                LLVM_ScalarOrVectorOf<AnyInteger>>;
def LLVM_FPExtOp : LLVM_CastOp<"fpext", "FPExt",
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>,
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>>;
def LLVM_FPTruncOp : LLVM_CastOp<"fptrunc", "FPTrunc",
                                 LLVM_ScalarOrVectorOf<LLVM_AnyFloat>,
                                 LLVM_ScalarOrVectorOf<LLVM_AnyFloat>>;

// Call-related operations.
def LLVM_InvokeOp : LLVM_Op<"invoke", [
                      AttrSizedOperandSegments,
                      DeclareOpInterfaceMethods<BranchOpInterface>,
                      DeclareOpInterfaceMethods<CallOpInterface>, Terminator]> {
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

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def LLVM_CallOp : LLVM_Op<"call",
                          [DeclareOpInterfaceMethods<FastmathFlagsInterface>,
                           DeclareOpInterfaceMethods<CallOpInterface>,
                           DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
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
                       DefaultValuedAttr<LLVM_FastmathFlagsAttr,
                                         "{}">:$fastmathFlags);
  let results = (outs Optional<LLVM_Type>:$result);

  let builders = [
    OpBuilder<(ins "LLVMFuncOp":$func, "ValueRange":$args)>,
    OpBuilder<(ins "TypeRange":$results, "StringAttr":$callee,
                   CArg<"ValueRange", "{}">:$args)>,
    OpBuilder<(ins "TypeRange":$results, "StringRef":$callee,
                   CArg<"ValueRange", "{}">:$args)>
  ];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// ExtractElementOp
//===----------------------------------------------------------------------===//

def LLVM_ExtractElementOp : LLVM_Op<"extractelement", [Pure,
    TypesMatchWith<"result type matches vector element type", "vector", "res",
                   "LLVM::getVectorElementType($_self)">]> {
  let summary = "Extract an element from an LLVM vector.";

  let arguments = (ins LLVM_AnyVector:$vector, AnyInteger:$position);
  let results = (outs LLVM_Type:$res);

  let builders = [
    OpBuilder<(ins "Value":$vector, "Value":$position,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let assemblyFormat = [{
    $vector `[` $position `:` type($position) `]` attr-dict `:` type($vector)
  }];

  string llvmInstName = "ExtractElement";
  string llvmBuilder = [{
    $res = builder.CreateExtractElement($vector, $position);
  }];
  string mlirBuilder = [{
    $res = $_builder.create<LLVM::ExtractElementOp>(
      $_location, $vector, $position);
  }];
}

//===----------------------------------------------------------------------===//
// ExtractValueOp
//===----------------------------------------------------------------------===//

def LLVM_ExtractValueOp : LLVM_Op<"extractvalue", [Pure]> {
  let summary = "Extract a value from an LLVM struct.";

  let arguments = (ins LLVM_AnyAggregate:$container, DenseI64ArrayAttr:$position);
  let results = (outs LLVM_Type:$res);

  let builders = [
    OpBuilder<(ins "Value":$container, "ArrayRef<int64_t>":$position)>
  ];

  let assemblyFormat = [{
    $container `` $position attr-dict `:` type($container)
    custom<InsertExtractValueElementType>(type($res), ref(type($container)),
                                          ref($position))
  }];

  let hasFolder = 1;
  let hasVerifier = 1;

  string llvmInstName = "ExtractValue";
  string llvmBuilder = [{
    $res = builder.CreateExtractValue($container, extractPosition($position));
  }];
  string mlirBuilder = [{
    auto *evInst = cast<llvm::ExtractValueInst>(inst);
    $res = $_builder.create<LLVM::ExtractValueOp>($_location,
      $container, getPositionFromIndices(evInst->getIndices()));
  }];
}

//===----------------------------------------------------------------------===//
// InsertElementOp
//===----------------------------------------------------------------------===//

def LLVM_InsertElementOp : LLVM_Op<"insertelement", [Pure,
    TypesMatchWith<"argument type matches vector element type", "vector",
                   "value", "LLVM::getVectorElementType($_self)">,
    AllTypesMatch<["res", "vector"]>]> {
  let summary = "Insert an element into an LLVM vector.";

  let arguments = (ins LLVM_AnyVector:$vector, LLVM_PrimitiveType:$value,
                       AnyInteger:$position);
  let results = (outs LLVM_AnyVector:$res);

  let builders = [LLVM_OneResultOpBuilder];

  let assemblyFormat = [{
    $value `,` $vector `[` $position `:` type($position) `]` attr-dict `:`
    type($vector)
  }];

  string llvmInstName = "InsertElement";
  string llvmBuilder = [{
    $res = builder.CreateInsertElement($vector, $value, $position);
  }];
  string mlirBuilder = [{
    $res = $_builder.create<LLVM::InsertElementOp>(
      $_location, $vector, $value, $position);
  }];
}

//===----------------------------------------------------------------------===//
// InsertValueOp
//===----------------------------------------------------------------------===//

def LLVM_InsertValueOp : LLVM_Op<
    "insertvalue", [Pure, AllTypesMatch<["container", "res"]>]> {
  let summary = "Insert a value into an LLVM struct.";

  let arguments = (ins LLVM_AnyAggregate:$container, LLVM_PrimitiveType:$value,
                       DenseI64ArrayAttr:$position);
  let results = (outs LLVM_AnyAggregate:$res);

  let assemblyFormat = [{
    $value `,` $container `` $position attr-dict `:` type($container)
    custom<InsertExtractValueElementType>(type($value), ref(type($container)),
                                          ref($position))
  }];

  let hasVerifier = 1;

  string llvmInstName = "InsertValue";
  string llvmBuilder = [{
    $res = builder.CreateInsertValue($container, $value,
                                     extractPosition($position));
  }];
  string mlirBuilder = [{
    auto *ivInst = cast<llvm::InsertValueInst>(inst);
    $res = $_builder.create<LLVM::InsertValueOp>($_location,
      $container, $value, getPositionFromIndices(ivInst->getIndices()));
  }];
}

//===----------------------------------------------------------------------===//
// ShuffleVectorOp
//===----------------------------------------------------------------------===//

def LLVM_ShuffleVectorOp : LLVM_Op<"shufflevector",
    [Pure, AllTypesMatch<["v1", "v2"]>]> {
  let summary = "Construct a permutation of two vectors.";

  let arguments = (ins LLVM_AnyVector:$v1, LLVM_AnyVector:$v2,
                       DenseI32ArrayAttr:$mask);
  let results = (outs LLVM_AnyVector:$res);

  let builders = [
    OpBuilder<(ins "Value":$v1, "Value":$v2, "DenseI32ArrayAttr":$mask,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins "Value":$v1, "Value":$v2, "ArrayRef<int32_t>":$mask)>
  ];

  let assemblyFormat = [{
    $v1 `,` $v2 $mask attr-dict `:` type($v1)
    custom<ShuffleType>(ref(type($v1)), type($res), ref($mask))
  }];

  let hasVerifier = 1;

  string llvmInstName = "ShuffleVector";
  string llvmBuilder = [{
    $res = builder.CreateShuffleVector($v1, $v2, $mask);
  }];
  string mlirBuilder = [{
    auto *svInst = cast<llvm::ShuffleVectorInst>(inst);
    SmallVector<int32_t> mask(svInst->getShuffleMask());
    $res = $_builder.create<LLVM::ShuffleVectorOp>(
      $_location, $v1, $v2, mask);
  }];
}

// Misc operations.
def LLVM_SelectOp
    : LLVM_Op<"select",
          [Pure, AllTypesMatch<["trueValue", "falseValue", "res"]>]>,
      LLVM_Builder<
          "$res = builder.CreateSelect($condition, $trueValue, $falseValue);"> {
  let arguments = (ins LLVM_ScalarOrVectorOf<I1>:$condition,
                   LLVM_Type:$trueValue, LLVM_Type:$falseValue);
  let results = (outs LLVM_Type:$res);
  let assemblyFormat = "operands attr-dict `:` type($condition) `,` type($res)";
  string llvmInstName = "Select";
  string mlirBuilder = [{
    $res = $_builder.create<LLVM::SelectOp>(
      $_location, $_resultType, $condition, $trueValue, $falseValue);
  }];
}
def LLVM_FreezeOp : LLVM_Op<"freeze", [SameOperandsAndResultType]> {
  let arguments = (ins LLVM_Type:$val);
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "$val attr-dict `:` type($val)";
  string llvmInstName = "Freeze";
  string llvmBuilder = "builder.CreateFreeze($val);";
  string mlirBuilder = [{
    $res = $_builder.create<LLVM::FreezeOp>($_location, $val);
  }];
}

// Terminators.
def LLVM_BrOp : LLVM_TerminatorOp<"br",
    [DeclareOpInterfaceMethods<BranchOpInterface>, Pure]> {
  let arguments = (ins Variadic<LLVM_Type>:$destOperands);
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
  let builders = [LLVM_TerminatorPassthroughOpBuilder];
}
def LLVM_CondBrOp : LLVM_TerminatorOp<"cond_br",
    [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BranchOpInterface>,
     Pure]> {
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

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def LLVM_ReturnOp : LLVM_TerminatorOp<"return", [Pure]> {
  let arguments = (ins Optional<LLVM_Type>:$arg);
  let assemblyFormat = "attr-dict ($arg^ `:` type($arg))?";

  let builders = [
    OpBuilder<(ins "ValueRange":$args), [{
      build($_builder, $_state, TypeRange(), args);
    }]>
  ];

  let hasVerifier = 1;

  string llvmInstName = "Ret";
  string llvmBuilder = [{
    if ($_numOperands != 0)
      builder.CreateRet($arg);
    else
      builder.CreateRetVoid();
  }];
  string mlirBuilder = [{
    $_builder.create<LLVM::ReturnOp>($_location, convertValues(llvmOperands));
  }];
}

def LLVM_ResumeOp : LLVM_TerminatorOp<"resume"> {
  let arguments = (ins LLVM_Type:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
  let hasVerifier = 1;
  string llvmInstName = "Resume";
  string llvmBuilder = [{ builder.CreateResume($value); }];
  string mlirBuilder = [{
    $_builder.create<LLVM::ResumeOp>($_location, $value);
  }];
}
def LLVM_UnreachableOp : LLVM_TerminatorOp<"unreachable"> {
  let assemblyFormat = "attr-dict";
  string llvmInstName = "Unreachable";
  string llvmBuilder = [{ builder.CreateUnreachable(); }];
  string mlirBuilder = [{
    $_builder.create<LLVM::UnreachableOp>($_location);
  }];
}

def LLVM_SwitchOp : LLVM_TerminatorOp<"switch",
    [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BranchOpInterface>,
     Pure]> {
  let arguments = (ins
    AnyInteger:$value,
    Variadic<AnyType>:$defaultOperands,
    VariadicOfVariadic<AnyType, "case_operand_segments">:$caseOperands,
    OptionalAttr<ElementsAttr>:$case_values,
    DenseI32ArrayAttr:$case_operand_segments,
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

def LLVM_AddressOfOp : LLVM_Op<"mlir.addressof",
    [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins FlatSymbolRefAttr:$global_name);
  let results = (outs LLVM_AnyPointer:$res);

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
            LLVM::LLVMPointerType::get(func.getFunctionType()), func.getName());
      $_state.addAttributes(attrs);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the llvm.mlir.global operation that defined the value referenced
    /// here.
    GlobalOp getGlobal(SymbolTableCollection &symbolTable);

    /// Return the llvm.func operation that is referenced here.
    LLVMFuncOp getFunction(SymbolTableCollection &symbolTable);
  }];

  let assemblyFormat = "$global_name attr-dict `:` qualified(type($res))";
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
    UnitAttr:$thread_local_,
    OptionalAttr<AnyAttr>:$value,
    OptionalAttr<I64Attr>:$alignment,
    DefaultValuedAttr<ConfinedAttr<I32Attr, [IntNonNegative]>, "0">:$addr_space,
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
      CArg<"bool", "false">:$thread_local_,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    /// Return the LLVM type of the global.
    Type getType() {
      return getGlobalType();
    }
    /// Return the initializer attribute if it exists, or a null attribute.
    Attribute getValueOrNull() {
      return getValue().value_or(Attribute());
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
  let hasRegionVerifier = 1;
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
    AutomaticAllocationScope, IsolatedFromAbove, FunctionOpInterface,
    CallableOpInterface, Symbol
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

  let arguments = (ins
    StrAttr:$sym_name,
    TypeAttrOf<LLVM_FunctionType>:$function_type,
    DefaultValuedAttr<Linkage, "Linkage::External">:$linkage,
    UnitAttr:$dso_local,
    DefaultValuedAttr<CConv, "CConv::C">:$CConv,
    OptionalAttr<FlatSymbolRefAttr>:$personality,
    OptionalAttr<StrAttr>:$garbageCollector,
    OptionalAttr<ArrayAttr>:$passthrough
  );

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "StringRef":$name, "Type":$type,
      CArg<"Linkage", "Linkage::External">:$linkage,
      CArg<"bool", "false">:$dsoLocal,
      CArg<"CConv", "CConv::C">:$cconv,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>
  ];

  let extraClassDeclaration = [{
    // Add an entry block to an empty function, and set up the block arguments
    // to match the signature of the function.
    Block *addEntryBlock();

    bool isVarArg() { return getFunctionType().isVarArg(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getParams(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getReturnTypes(); }

    /// Returns the callable region, which is the function body.
    Region *getCallableRegion() { return &getBody(); }

    /// Returns the callable result type, which is the function return type.
    ArrayRef<Type> getCallableResults() { return getFunctionType().getReturnTypes(); }

  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def LLVM_NullOp
    : LLVM_Op<"mlir.null", [Pure]>,
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
  let assemblyFormat = "attr-dict `:` qualified(type($res))";
}

def LLVM_UndefOp : LLVM_Op<"mlir.undef", [Pure]>,
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
    : LLVM_Op<"mlir.constant", [Pure, ConstantLike]>,
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

  let assemblyFormat = "`(` $value `)` attr-dict `:` type($res)";

  let builders = [
    LLVM_OneResultOpBuilder,
    OpBuilder<(ins "Type":$type, "int64_t":$value), [{
      build($_builder, $_state, type, $_builder.getIntegerAttr(type, value));
    }]>,
    OpBuilder<(ins "Type":$type, "const APInt &":$value), [{
      build($_builder, $_state, type, $_builder.getIntegerAttr(type, value));
    }]>,
    OpBuilder<(ins "Type":$type, "const APFloat &":$value), [{
      build($_builder, $_state, type, $_builder.getFloatAttr(type, value));
    }]>,
    OpBuilder<(ins "TypedAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>
  ];

  let hasFolder = 1;
  let hasVerifier = 1;
}

// Atomic operations.
//

def LLVM_AtomicRMWType : AnyTypeOf<[LLVM_AnyFloat, AnyInteger]>;

// FIXME: Need to add alignment and syncscope attribute to MLIR atomicrmw
// operation.
def LLVM_AtomicRMWOp : LLVM_Op<"atomicrmw"> {
  let arguments = (ins AtomicBinOp:$bin_op,
                   LLVM_PointerTo<LLVM_AtomicRMWType>:$ptr,
                   LLVM_AtomicRMWType:$val, AtomicOrdering:$ordering);
  let results = (outs LLVM_AtomicRMWType:$res);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  string llvmInstName = "AtomicRMW";
  string llvmBuilder = [{
    $res = builder.CreateAtomicRMW(getLLVMAtomicBinOp($bin_op), $ptr, $val,
                                   llvm::MaybeAlign(),
                                   getLLVMAtomicOrdering($ordering));
  }];
  string mlirBuilder = [{
    auto *atomicInst = cast<llvm::AtomicRMWInst>(inst);
    $res = $_builder.create<LLVM::AtomicRMWOp>($_location, $_resultType,
        getLLVMAtomicBinOp(atomicInst->getOperation()), $ptr, $val,
        getLLVMAtomicOrdering(atomicInst->getOrdering()));
  }];
  // Only $ptr and $val are llvm instruction operands.
  list<int> llvmArgIndices = [-1, 0, 1, -1];
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
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  string llvmInstName = "AtomicCmpXchg";
  string llvmBuilder = [{
    $res = builder.CreateAtomicCmpXchg($ptr, $cmp, $val, llvm::MaybeAlign(),
                   getLLVMAtomicOrdering($success_ordering),
                   getLLVMAtomicOrdering($failure_ordering));
  }];
  string mlirBuilder = [{
    auto *cmpXchgInst = cast<llvm::AtomicCmpXchgInst>(inst);
    $res = $_builder.create<LLVM::AtomicCmpXchgOp>(
      $_location, $_resultType, $ptr, $cmp, $val,
      getLLVMAtomicOrdering(cmpXchgInst->getSuccessOrdering()),
      getLLVMAtomicOrdering(cmpXchgInst->getFailureOrdering()));
  }];
}

def LLVM_FenceOp : LLVM_Op<"fence"> {
  let arguments = (ins AtomicOrdering:$ordering, StrAttr:$syncscope);
  let builders = [LLVM_VoidResultTypeOpBuilder, LLVM_ZeroResultOpBuilder];
  string llvmInstName = "Fence";
  let llvmBuilder = [{
    llvm::LLVMContext &llvmContext = builder.getContext();
    builder.CreateFence(getLLVMAtomicOrdering($ordering),
      llvmContext.getOrInsertSyncScopeID($syncscope));
  }];
  string mlirBuilder = [{
    llvm::FenceInst *fenceInst = cast<llvm::FenceInst>(inst);
    $_builder.create<LLVM::FenceOp>(
      $_location,
      getLLVMAtomicOrdering(fenceInst->getOrdering()),
      getLLVMSyncScope(fenceInst));
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
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
