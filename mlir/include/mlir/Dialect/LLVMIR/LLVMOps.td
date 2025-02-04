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
include "mlir/Dialect/LLVMIR/LLVMEnums.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

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
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
  string llvmInstName = instName;
}
class LLVM_IntArithmeticOp<string mnemonic, string instName,
                           list<Trait> traits = []> :
    LLVM_ArithmeticOpBase<AnySignlessInteger, mnemonic, instName, traits> {
  let arguments = commonArgs;
  string mlirBuilder = [{
    $res = $_builder.create<$_qualCppClassName>($_location, $lhs, $rhs);
  }];
}
class LLVM_IntArithmeticOpWithOverflowFlag<string mnemonic, string instName,
                                   list<Trait> traits = []> :
    LLVM_ArithmeticOpBase<AnySignlessInteger, mnemonic, instName,
    !listconcat([DeclareOpInterfaceMethods<IntegerOverflowFlagsInterface>], traits)> {
  dag iofArg = (ins EnumProp<"IntegerOverflowFlags", "", "IntegerOverflowFlags::none">:$overflowFlags);
  let arguments = !con(commonArgs, iofArg);

  string mlirBuilder = [{
    auto op = $_builder.create<$_qualCppClassName>($_location, $lhs, $rhs);
    moduleImport.setIntegerOverflowFlags(inst, op);
    $res = op;
  }];
  let assemblyFormat = [{
    $lhs `,` $rhs `` custom<OverflowFlags>($overflowFlags) attr-dict `:` type($res)
  }];
  string llvmBuilder =
    "$res = builder.Create" # instName #
    "($lhs, $rhs, /*Name=*/\"\", op.hasNoUnsignedWrap(), op.hasNoSignedWrap());";
}
class LLVM_IntArithmeticOpWithExactFlag<string mnemonic, string instName,
                                   list<Trait> traits = []> :
    LLVM_ArithmeticOpBase<AnySignlessInteger, mnemonic, instName,
    !listconcat([DeclareOpInterfaceMethods<ExactFlagInterface>], traits)> {
  let arguments = !con(commonArgs, (ins UnitAttr:$isExact));

  string mlirBuilder = [{
    auto op = $_builder.create<$_qualCppClassName>($_location, $lhs, $rhs);
    moduleImport.setExactFlag(inst, op);
    $res = op;
  }];
  let assemblyFormat = [{
    (`exact` $isExact^)? $lhs `,` $rhs attr-dict `:` type($res)
  }];
  string llvmBuilder =
    "$res = builder.Create" # instName #
    "($lhs, $rhs, /*Name=*/\"\", op.getIsExact());";
}
class LLVM_IntArithmeticOpWithDisjointFlag<string mnemonic, string instName,
                                   list<Trait> traits = []> :
    LLVM_ArithmeticOpBase<AnySignlessInteger, mnemonic, instName,
    !listconcat([DeclareOpInterfaceMethods<DisjointFlagInterface>], traits)> {
  let arguments = !con(commonArgs, (ins UnitAttr:$isDisjoint));

  string mlirBuilder = [{
    auto op = $_builder.create<$_qualCppClassName>($_location, $lhs, $rhs);
    moduleImport.setDisjointFlag(inst, op);
    $res = op;
  }];
  let assemblyFormat = [{
    (`disjoint` $isDisjoint^)? $lhs `,` $rhs attr-dict `:` type($res)
  }];
  string llvmBuilder = [{
    auto inst = builder.Create}] # instName # [{($lhs, $rhs, /*Name=*/"");
    moduleTranslation.setDisjointFlag(op, inst);
    $res = inst;
  }];
}
class LLVM_FloatArithmeticOp<string mnemonic, string instName,
                             list<Trait> traits = []> :
    LLVM_ArithmeticOpBase<LLVM_AnyFloat, mnemonic, instName,
    !listconcat([DeclareOpInterfaceMethods<FastmathFlagsInterface>], traits)> {
  dag fmfArg = (
    ins DefaultValuedAttr<LLVM_FastmathFlagsAttr, "{}">:$fastmathFlags);
  let arguments = !con(commonArgs, fmfArg);
  string mlirBuilder = [{
    auto op = $_builder.create<$_qualCppClassName>($_location, $lhs, $rhs);
    moduleImport.setFastmathFlagsAttr(inst, op);
    $res = op;
  }];
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
  let assemblyFormat = "$operand attr-dict `:` type($res)";
  string llvmInstName = instName;
  string mlirBuilder = [{
    auto op = $_builder.create<$_qualCppClassName>($_location, $operand);
    moduleImport.setFastmathFlagsAttr(inst, op);
    $res = op;
   }];
}

// Integer binary operations.
def LLVM_AddOp : LLVM_IntArithmeticOpWithOverflowFlag<"add", "Add",
    [Commutative]>;
def LLVM_SubOp : LLVM_IntArithmeticOpWithOverflowFlag<"sub", "Sub", []>;
def LLVM_MulOp : LLVM_IntArithmeticOpWithOverflowFlag<"mul", "Mul",
    [Commutative]>;
def LLVM_UDivOp : LLVM_IntArithmeticOpWithExactFlag<"udiv", "UDiv">;
def LLVM_SDivOp : LLVM_IntArithmeticOpWithExactFlag<"sdiv", "SDiv">;
def LLVM_URemOp : LLVM_IntArithmeticOp<"urem", "URem">;
def LLVM_SRemOp : LLVM_IntArithmeticOp<"srem", "SRem">;
def LLVM_AndOp : LLVM_IntArithmeticOp<"and", "And">;
def LLVM_OrOp : LLVM_IntArithmeticOpWithDisjointFlag<"or", "Or"> {
  let hasFolder = 1;
}
def LLVM_XOrOp : LLVM_IntArithmeticOp<"xor", "Xor">;
def LLVM_ShlOp : LLVM_IntArithmeticOpWithOverflowFlag<"shl", "Shl", []> {
  let hasFolder = 1;
}
def LLVM_LShrOp : LLVM_IntArithmeticOpWithExactFlag<"lshr", "LShr">;
def LLVM_AShrOp : LLVM_IntArithmeticOpWithExactFlag<"ashr", "AShr">;

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
                   AnyTypeOf<[LLVM_ScalarOrVectorOf<AnySignlessInteger>,
                              LLVM_ScalarOrVectorOf<LLVM_AnyPointer>]>:$lhs,
                   AnyTypeOf<[LLVM_ScalarOrVectorOf<AnySignlessInteger>,
                              LLVM_ScalarOrVectorOf<LLVM_AnyPointer>]>:$rhs);
  let hasCustomAssemblyFormat = 1;
  string llvmInstName = "ICmp";
  string llvmBuilder = [{
    $res = builder.CreateICmp(
            convertICmpPredicateToLLVM($predicate), $lhs, $rhs);
  }];
  string mlirBuilder = [{
    auto *iCmpInst = cast<llvm::ICmpInst>(inst);
    $res = $_builder.create<$_qualCppClassName>($_location,
            convertICmpPredicateFromLLVM(iCmpInst->getPredicate()), $lhs, $rhs);
  }];
  // Set the $predicate index to -1 to indicate there is no matching operand
  // and decrement the following indices.
  list<int> llvmArgIndices = [-1, 0, 1];
  let hasFolder = 1;
}

// Other floating-point operations.
def LLVM_FCmpOp : LLVM_ArithmeticCmpOp<"fcmp", [
    Pure, DeclareOpInterfaceMethods<FastmathFlagsInterface>]> {
  let arguments = (ins FCmpPredicate:$predicate,
                   LLVM_ScalarOrVectorOf<LLVM_AnyFloat>:$lhs,
                   LLVM_ScalarOrVectorOf<LLVM_AnyFloat>:$rhs,
                   DefaultValuedAttr<LLVM_FastmathFlagsAttr,
                                     "{}">:$fastmathFlags);
  let hasCustomAssemblyFormat = 1;
  string llvmInstName = "FCmp";
  string llvmBuilder = [{
    $res = builder.CreateFCmp(convertFCmpPredicateToLLVM($predicate), $lhs, $rhs);
  }];
  string mlirBuilder = [{
    auto *fCmpInst = cast<llvm::FCmpInst>(inst);
    auto op = $_builder.create<$_qualCppClassName>(
      $_location, convertFCmpPredicateFromLLVM(fCmpInst->getPredicate()), $lhs, $rhs);
    moduleImport.setFastmathFlagsAttr(inst, op);
    $res = op;
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

// Memory-related operations.
def LLVM_AllocaOp : LLVM_Op<"alloca",
    [DeclareOpInterfaceMethods<PromotableAllocationOpInterface>,
     DeclareOpInterfaceMethods<DestructurableAllocationOpInterface>]>,
  LLVM_MemOpPatterns {
  let arguments = (ins AnySignlessInteger:$arraySize,
                   OptionalAttr<I64Attr>:$alignment,
                   TypeAttr:$elem_type,
                   UnitAttr:$inalloca);
  let results = (outs Res<LLVM_AnyPointer, "",
                          [MemAlloc<AutomaticAllocationScopeResource>]>:$res);
  string llvmInstName = "Alloca";
  string llvmBuilder = [{
    auto addrSpace = $_resultType->getPointerAddressSpace();
    llvm::Type *elementType = moduleTranslation.convertType($elem_type);
    auto *inst = builder.CreateAlloca(elementType, addrSpace, $arraySize);
    }] # setAlignmentCode # [{
    inst->setUsedWithInAlloca($inalloca);
    $res = inst;
  }];
  string mlirBuilder = [{
    auto *allocaInst = cast<llvm::AllocaInst>(inst);
    Type allocatedType =
      moduleImport.convertType(allocaInst->getAllocatedType());
    unsigned alignment = allocaInst->getAlign().value();
    $res = $_builder.create<LLVM::AllocaOp>(
      $_location, $_resultType, $arraySize,
      alignment == 0 ? IntegerAttr() : $_builder.getI64IntegerAttr(alignment),
      allocatedType, allocaInst->isUsedWithInAlloca());
  }];
  let builders = [
    OpBuilder<(ins "Type":$resultType, "Type":$elementType, "Value":$arraySize,
               CArg<"unsigned", "0">:$alignment),
    [{
      build($_builder, $_state, resultType, arraySize,
            alignment == 0 ? IntegerAttr()
                           : $_builder.getI64IntegerAttr(alignment),
            elementType, false);

    }]>
    ];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LLVM_GEPOp : LLVM_Op<"getelementptr", [Pure,
    DeclareOpInterfaceMethods<PromotableOpInterface>,
    DeclareOpInterfaceMethods<SafeMemorySlotAccessOpInterface>,
    DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>,
    DeclareOpInterfaceMethods<ViewLikeOpInterface>
    ]> {
  let arguments = (ins LLVM_ScalarOrVectorOf<LLVM_AnyPointer>:$base,
                   Variadic<LLVM_ScalarOrVectorOf<AnySignlessInteger>>:$dynamicIndices,
                   DenseI32ArrayAttr:$rawConstantIndices,
                   TypeAttr:$elem_type,
                   UnitAttr:$inbounds);
  let results = (outs LLVM_ScalarOrVectorOf<LLVM_AnyPointer>:$res);
  let skipDefaultBuilders = 1;

  let description = [{
    This operation mirrors LLVM IRs 'getelementptr' operation that is used to
    perform pointer arithmetic.

    Like in LLVM IR, it is possible to use both constants as well as SSA values
    as indices. In the case of indexing within a structure, it is required to
    either use constant indices directly, or supply a constant SSA value.

    An optional 'inbounds' attribute specifies the low-level pointer arithmetic
    overflow behavior that LLVM uses after lowering the operation to LLVM IR.

    Examples:

    ```mlir
    // GEP with an SSA value offset
    %0 = llvm.getelementptr %1[%2] : (!llvm.ptr, i64) -> !llvm.ptr, f32

    // GEP with a constant offset and the inbounds attribute set
    %0 = llvm.getelementptr inbounds %1[3] : (!llvm.ptr) -> !llvm.ptr, f32

    // GEP with constant offsets into a structure
    %0 = llvm.getelementptr %1[0, 1]
       : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, f32)>
    ```
  }];

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Type":$elementType, "Value":$basePtr,
               "ValueRange":$indices, CArg<"bool", "false">:$inbounds,
               CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
    OpBuilder<(ins "Type":$resultType, "Type":$elementType, "Value":$basePtr,
               "ArrayRef<GEPArg>":$indices, CArg<"bool", "false">:$inbounds,
               CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes)>,
  ];
  let llvmBuilder = [{
    SmallVector<llvm::Value *> indices;
    indices.reserve($rawConstantIndices.size());
    GEPIndicesAdaptor<decltype($dynamicIndices)>
        gepIndices(op.getRawConstantIndicesAttr(), $dynamicIndices);
    for (PointerUnion<IntegerAttr, llvm::Value*> valueOrAttr : gepIndices) {
      if (llvm::Value* value = ::llvm::dyn_cast<llvm::Value*>(valueOrAttr))
        indices.push_back(value);
      else
        indices.push_back(
            builder.getInt32(cast<IntegerAttr>(valueOrAttr).getInt()));
    }
    Type baseElementType = op.getElemType();
    llvm::Type *elementType = moduleTranslation.convertType(baseElementType);
    $res = builder.CreateGEP(elementType, $base, indices, "", $inbounds);
  }];
  let assemblyFormat = [{
    (`inbounds` $inbounds^)?
    $base `[` custom<GEPIndices>($dynamicIndices, $rawConstantIndices) `]` attr-dict
    `:` functional-type(operands, results) `,` $elem_type
  }];

  let extraClassDeclaration = [{
    constexpr static int32_t kDynamicIndex = std::numeric_limits<int32_t>::min();

    GEPIndicesAdaptor<ValueRange> getIndices();
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

def LLVM_LoadOp : LLVM_MemAccessOpBase<"load",
    [DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<PromotableMemOpInterface>,
     DeclareOpInterfaceMethods<SafeMemorySlotAccessOpInterface>]> {
  dag args = (ins LLVM_AnyPointer:$addr,
              OptionalAttr<I64Attr>:$alignment,
              UnitAttr:$volatile_,
              UnitAttr:$nontemporal,
              UnitAttr:$invariant,
              UnitAttr:$invariantGroup,
              DefaultValuedAttr<
                AtomicOrdering, "AtomicOrdering::not_atomic">:$ordering,
              OptionalAttr<StrAttr>:$syncscope);
  // Append the aliasing related attributes defined in LLVM_MemAccessOpBase.
  let arguments = !con(args, aliasAttrs);
  let results = (outs LLVM_LoadableType:$res);
  string llvmInstName = "Load";
  let description = [{
    The `load` operation is used to read from memory. A load may be marked as
    atomic, volatile, and/or nontemporal, and takes a number of optional
    attributes that specify aliasing information.

    An atomic load only supports a limited set of pointer, integer, and
    floating point types, and requires an explicit alignment.

    Examples:
    ```mlir
    // A volatile load of a float variable.
    %0 = llvm.load volatile %ptr : !llvm.ptr -> f32

    // A nontemporal load of a float variable.
    %0 = llvm.load %ptr {nontemporal} : !llvm.ptr -> f32

    // An atomic load of an integer variable.
    %0 = llvm.load %ptr atomic monotonic {alignment = 8 : i64}
        : !llvm.ptr -> i64
    ```

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#load-instruction
  }];
  let assemblyFormat = [{
    (`volatile` $volatile_^)? $addr
    (`atomic` (`syncscope` `(` $syncscope^ `)`)? $ordering^)?
    (`invariant` $invariant^)?
    (`invariant_group` $invariantGroup^)?
    attr-dict `:` qualified(type($addr)) `->` type($res)
  }];
  string llvmBuilder = [{
    auto *inst = builder.CreateLoad($_resultType, $addr, $volatile_);
    $res = inst;
    if ($invariant) {
      llvm::MDNode *metadata = llvm::MDNode::get(inst->getContext(), std::nullopt);
      inst->setMetadata(llvm::LLVMContext::MD_invariant_load, metadata);
    }
  }] # setOrderingCode
     # setSyncScopeCode
     # setAlignmentCode
     # setNonTemporalMetadataCode
     # setInvariantGroupCode
     # setAccessGroupsMetadataCode
     # setAliasAnalysisMetadataCode;
  string mlirBuilder = [{
    auto *loadInst = cast<llvm::LoadInst>(inst);
    unsigned alignment = loadInst->getAlign().value();
    $res = $_builder.create<LLVM::LoadOp>($_location, $_resultType, $addr,
        alignment, loadInst->isVolatile(),
        loadInst->hasMetadata(llvm::LLVMContext::MD_nontemporal),
        loadInst->hasMetadata(llvm::LLVMContext::MD_invariant_load),
        loadInst->hasMetadata(llvm::LLVMContext::MD_invariant_group),
        convertAtomicOrderingFromLLVM(loadInst->getOrdering()),
        getLLVMSyncScope(loadInst));
  }];
  let builders = [
    OpBuilder<(ins "Type":$type, "Value":$addr,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isVolatile,
      CArg<"bool", "false">:$isNonTemporal, CArg<"bool", "false">:$isInvariant,
      CArg<"bool", "false">:$isInvariantGroup,
      CArg<"AtomicOrdering", "AtomicOrdering::not_atomic">:$ordering,
      CArg<"StringRef", "StringRef()">:$syncscope)>
  ];
  let hasVerifier = 1;
}

def LLVM_StoreOp : LLVM_MemAccessOpBase<"store",
    [DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<PromotableMemOpInterface>,
     DeclareOpInterfaceMethods<SafeMemorySlotAccessOpInterface>]> {
  dag args = (ins LLVM_LoadableType:$value,
              LLVM_AnyPointer:$addr,
              OptionalAttr<I64Attr>:$alignment,
              UnitAttr:$volatile_,
              UnitAttr:$nontemporal,
              UnitAttr:$invariantGroup,
              DefaultValuedAttr<
                AtomicOrdering, "AtomicOrdering::not_atomic">:$ordering,
              OptionalAttr<StrAttr>:$syncscope);
  // Append the aliasing related attributes defined in LLVM_MemAccessOpBase.
  let arguments = !con(args, aliasAttrs);
  string llvmInstName = "Store";
  let description = [{
    The `store` operation is used to write to memory. A store may be marked as
    atomic, volatile, and/or nontemporal, and takes a number of optional
    attributes that specify aliasing information.

    An atomic store only supports a limited set of pointer, integer, and
    floating point types, and requires an explicit alignment.

    Examples:
    ```mlir
    // A volatile store of a float variable.
    llvm.store volatile %val, %ptr : f32, !llvm.ptr

    // A nontemporal store of a float variable.
    llvm.store %val, %ptr {nontemporal} : f32, !llvm.ptr

    // An atomic store of an integer variable.
    llvm.store %val, %ptr atomic monotonic {alignment = 8 : i64}
        : i64, !llvm.ptr
    ```

    See the following link for more details:
    https://llvm.org/docs/LangRef.html#store-instruction
  }];
  let assemblyFormat = [{
    (`volatile` $volatile_^)? $value `,` $addr
    (`atomic` (`syncscope` `(` $syncscope^ `)`)? $ordering^)?
    (`invariant_group` $invariantGroup^)?
    attr-dict `:` type($value) `,` qualified(type($addr))
  }];
  string llvmBuilder = [{
    auto *inst = builder.CreateStore($value, $addr, $volatile_);
  }] # setOrderingCode
     # setSyncScopeCode
     # setAlignmentCode
     # setNonTemporalMetadataCode
     # setInvariantGroupCode
     # setAccessGroupsMetadataCode
     # setAliasAnalysisMetadataCode;
  string mlirBuilder = [{
    auto *storeInst = cast<llvm::StoreInst>(inst);
    unsigned alignment = storeInst->getAlign().value();
    $_op = $_builder.create<LLVM::StoreOp>($_location, $value, $addr,
        alignment, storeInst->isVolatile(),
        storeInst->hasMetadata(llvm::LLVMContext::MD_nontemporal),
        storeInst->hasMetadata(llvm::LLVMContext::MD_invariant_group),
        convertAtomicOrderingFromLLVM(storeInst->getOrdering()),
        getLLVMSyncScope(storeInst));
  }];
  let builders = [
    OpBuilder<(ins "Value":$value, "Value":$addr,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isVolatile,
      CArg<"bool", "false">:$isNonTemporal,
      CArg<"bool", "false">:$isInvariantGroup,
      CArg<"AtomicOrdering", "AtomicOrdering::not_atomic">:$ordering,
      CArg<"StringRef", "StringRef()">:$syncscope)>
  ];
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
class LLVM_CastOpWithNNegFlag<string mnemonic, string instName, Type type,
                  Type resultType, list<Trait> traits = []> :
    LLVM_Op<mnemonic, !listconcat([Pure], [DeclareOpInterfaceMethods<NonNegFlagInterface>], traits)>,
    LLVM_Builder<"$res = builder.Create" # instName # "($arg, $_resultType, /*Name=*/\"\", op.getNonNeg());"> {
  let arguments = (ins type:$arg, UnitAttr:$nonNeg);
  let results = (outs resultType:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "(`nneg` $nonNeg^)? $arg attr-dict `:` type($arg) `to` type($res)";
  string llvmInstName = instName;
  string mlirBuilder = [{
    auto op = $_builder.create<$_qualCppClassName>(
      $_location, $_resultType, $arg);
    moduleImport.setNonNegFlag(inst, op);
    $res = op;
  }];
}

class LLVM_CastOpWithOverflowFlag<string mnemonic, string instName, Type type,
                  Type resultType, list<Trait> traits = []> :
    LLVM_Op<mnemonic, !listconcat([Pure], [DeclareOpInterfaceMethods<IntegerOverflowFlagsInterface>], traits)>,
    LLVM_Builder<"$res = builder.Create" # instName # "($arg, $_resultType, /*Name=*/\"\", op.hasNoUnsignedWrap(), op.hasNoSignedWrap());"> {
  let arguments = (ins type:$arg, EnumProp<"IntegerOverflowFlags", "", "IntegerOverflowFlags::none">:$overflowFlags);
  let results = (outs resultType:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "$arg `` custom<OverflowFlags>($overflowFlags) attr-dict `:` type($arg) `to` type($res)";
  string llvmInstName = instName;
  string mlirBuilder = [{
    auto op = $_builder.create<$_qualCppClassName>(
      $_location, $_resultType, $arg);
    moduleImport.setIntegerOverflowFlags(inst, op);
    $res = op;
  }];
}

def LLVM_BitcastOp : LLVM_CastOp<"bitcast", "BitCast", LLVM_AnyNonAggregate,
    LLVM_AnyNonAggregate, [DeclareOpInterfaceMethods<PromotableOpInterface>]> {
  let hasFolder = 1;
  let hasVerifier = 1;
}
def LLVM_AddrSpaceCastOp : LLVM_CastOp<"addrspacecast", "AddrSpaceCast",
    LLVM_ScalarOrVectorOf<LLVM_AnyPointer>,
    LLVM_ScalarOrVectorOf<LLVM_AnyPointer>,
    [DeclareOpInterfaceMethods<PromotableOpInterface>,
     DeclareOpInterfaceMethods<ViewLikeOpInterface>]> {
  let hasFolder = 1;
}
def LLVM_IntToPtrOp : LLVM_CastOp<"inttoptr", "IntToPtr",
                                  LLVM_ScalarOrVectorOf<AnySignlessInteger>,
                                  LLVM_ScalarOrVectorOf<LLVM_AnyPointer>>;
def LLVM_PtrToIntOp : LLVM_CastOp<"ptrtoint", "PtrToInt",
                                  LLVM_ScalarOrVectorOf<LLVM_AnyPointer>,
                                  LLVM_ScalarOrVectorOf<AnySignlessInteger>>;
def LLVM_SExtOp : LLVM_CastOp<"sext", "SExt",
                              LLVM_ScalarOrVectorOf<AnySignlessInteger>,
                              LLVM_ScalarOrVectorOf<AnySignlessInteger>> {
  let hasVerifier = 1;
}
def LLVM_ZExtOp : LLVM_CastOpWithNNegFlag<"zext", "ZExt",
                              LLVM_ScalarOrVectorOf<AnySignlessInteger>,
                              LLVM_ScalarOrVectorOf<AnySignlessInteger>> {
  let hasFolder = 1;
  let hasVerifier = 1;
}
def LLVM_TruncOp : LLVM_CastOpWithOverflowFlag<"trunc", "Trunc",
                               LLVM_ScalarOrVectorOf<AnySignlessInteger>,
                               LLVM_ScalarOrVectorOf<AnySignlessInteger>>;
def LLVM_SIToFPOp : LLVM_CastOp<"sitofp", "SIToFP",
                                LLVM_ScalarOrVectorOf<AnySignlessInteger>,
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>>;
def LLVM_UIToFPOp : LLVM_CastOpWithNNegFlag<"uitofp", "UIToFP",
                                LLVM_ScalarOrVectorOf<AnySignlessInteger>,
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>>;
def LLVM_FPToSIOp : LLVM_CastOp<"fptosi", "FPToSI",
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>,
                                LLVM_ScalarOrVectorOf<AnySignlessInteger>>;
def LLVM_FPToUIOp : LLVM_CastOp<"fptoui", "FPToUI",
                                LLVM_ScalarOrVectorOf<LLVM_AnyFloat>,
                                LLVM_ScalarOrVectorOf<AnySignlessInteger>>;
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
                      DeclareOpInterfaceMethods<CallOpInterface>,
                      DeclareOpInterfaceMethods<BranchWeightOpInterface>,
                      Terminator]> {
  let arguments = (ins
                   OptionalAttr<TypeAttrOf<LLVM_FunctionType>>:$var_callee_type,
                   OptionalAttr<FlatSymbolRefAttr>:$callee,
                   Variadic<LLVM_Type>:$callee_operands,
                   OptionalAttr<DictArrayAttr>:$arg_attrs,
                   OptionalAttr<DictArrayAttr>:$res_attrs,
                   Variadic<LLVM_Type>:$normalDestOperands,
                   Variadic<LLVM_Type>:$unwindDestOperands,
                   OptionalAttr<DenseI32ArrayAttr>:$branch_weights,
                   DefaultValuedAttr<CConv, "CConv::C">:$CConv,
                   VariadicOfVariadic<LLVM_Type,
                                      "op_bundle_sizes">:$op_bundle_operands,
                   DenseI32ArrayAttr:$op_bundle_sizes,
                   OptionalAttr<ArrayAttr>:$op_bundle_tags);
  let results = (outs Optional<LLVM_Type>:$result);
  let successors = (successor AnySuccessor:$normalDest,
                              AnySuccessor:$unwindDest);

  let builders = [
    OpBuilder<(ins "LLVMFuncOp":$func,
      "ValueRange":$ops, "Block*":$normal, "ValueRange":$normalOps,
      "Block*":$unwind, "ValueRange":$unwindOps)>,
    OpBuilder<(ins "TypeRange":$tys, "FlatSymbolRefAttr":$callee,
      "ValueRange":$ops, "Block*":$normal, "ValueRange":$normalOps,
      "Block*":$unwind, "ValueRange":$unwindOps)>,
    OpBuilder<(ins "LLVMFunctionType":$calleeType, "FlatSymbolRefAttr":$callee,
      "ValueRange":$ops, "Block*":$normal, "ValueRange":$normalOps,
      "Block*":$unwind, "ValueRange":$unwindOps)>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    /// Returns the callee function type.
    LLVMFunctionType getCalleeFunctionType();
  }];
}

def LLVM_LandingpadOp : LLVM_Op<"landingpad"> {
  let arguments = (ins UnitAttr:$cleanup, Variadic<LLVM_Type>);
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// VAArgOp
//===----------------------------------------------------------------------===//

def LLVM_VaArgOp : LLVM_Op<"va_arg"> {
  let arguments = (ins LLVM_AnyPointer:$arg);
  let results = (outs LLVM_Type:$res);

  let builders = [LLVM_OneResultOpBuilder];

  let llvmBuilder = " $res = builder.CreateVAArg($arg, $_resultType); ";
  let assemblyFormat = "$arg attr-dict `:` functional-type($arg, $res)";

  string llvmInstName = "VAArg";
  string mlirBuilder = [{
    $res = $_builder.create<mlir::LLVM::VaArgOp>(
      $_location, $_resultType, $arg);
  }];
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def LLVM_CallOp : LLVM_MemAccessOpBase<"call",
                    [AttrSizedOperandSegments,
                     DeclareOpInterfaceMethods<FastmathFlagsInterface>,
                     DeclareOpInterfaceMethods<CallOpInterface>,
                     DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                     DeclareOpInterfaceMethods<BranchWeightOpInterface>]> {
  let summary = "Call to an LLVM function.";
  let description = [{
    In LLVM IR, functions may return either 0 or 1 value. LLVM IR dialect
    implements this behavior by providing a variadic `call` operation for 0- and
    1-result functions. Even though MLIR supports multi-result functions, LLVM
    IR dialect disallows them.

    The `call` instruction supports both direct and indirect calls. Direct calls
    start with a function name (`@`-prefixed) and indirect calls start with an
    SSA value (`%`-prefixed). The direct callee, if present, is stored as a
    function attribute `callee`. For indirect calls, the callee is of `!llvm.ptr` type
    and is stored as the first value in `callee_operands`. If and only if the
    callee is a variadic function, the `var_callee_type` attribute must carry
    the variadic LLVM function type. The trailing type list contains the
    optional indirect callee type and the MLIR function type, which differs from
    the LLVM function type that uses an explicit void type to model functions
    that do not return a value.

    Examples:

    ```mlir
    // Direct call without arguments and with one result.
    %0 = llvm.call @foo() : () -> (f32)

    // Direct call with arguments and without a result.
    llvm.call @bar(%0) : (f32) -> ()

    // Indirect call with an argument and without a result.
    %1 = llvm.mlir.addressof @foo : !llvm.ptr
    llvm.call %1(%0) : !llvm.ptr, (f32) -> ()

    // Direct variadic call.
    llvm.call @printf(%0, %1) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32

    // Indirect variadic call
    llvm.call %1(%0) vararg(!llvm.func<void (...)>) : !llvm.ptr, (i32) -> ()
    ```
  }];

  dag args = (ins OptionalAttr<TypeAttrOf<LLVM_FunctionType>>:$var_callee_type,
                  OptionalAttr<FlatSymbolRefAttr>:$callee,
                  Variadic<LLVM_Type>:$callee_operands,
                  DefaultValuedAttr<LLVM_FastmathFlagsAttr,
                                   "{}">:$fastmathFlags,
                  OptionalAttr<DenseI32ArrayAttr>:$branch_weights,
                  DefaultValuedAttr<CConv, "CConv::C">:$CConv,
                  DefaultValuedAttr<TailCallKind, "TailCallKind::None">:$TailCallKind,
                  OptionalAttr<LLVM_MemoryEffectsAttr>:$memory_effects,
                  OptionalAttr<UnitAttr>:$convergent,
                  OptionalAttr<UnitAttr>:$no_unwind,
                  OptionalAttr<UnitAttr>:$will_return,
                  VariadicOfVariadic<LLVM_Type,
                                     "op_bundle_sizes">:$op_bundle_operands,
                  DenseI32ArrayAttr:$op_bundle_sizes,
                  OptionalAttr<ArrayAttr>:$op_bundle_tags,
                  OptionalAttr<DictArrayAttr>:$arg_attrs,
                  OptionalAttr<DictArrayAttr>:$res_attrs);
  // Append the aliasing related attributes defined in LLVM_MemAccessOpBase.
  let arguments = !con(args, aliasAttrs);
  let results = (outs Optional<LLVM_Type>:$result);
  let builders = [
    OpBuilder<(ins "LLVMFuncOp":$func, "ValueRange":$args)>,
    OpBuilder<(ins "LLVMFunctionType":$calleeType, "ValueRange":$args)>,
    OpBuilder<(ins "TypeRange":$results, "StringAttr":$callee,
                   CArg<"ValueRange", "{}">:$args)>,
    OpBuilder<(ins "TypeRange":$results, "FlatSymbolRefAttr":$callee,
                   CArg<"ValueRange", "{}">:$args)>,
    OpBuilder<(ins "TypeRange":$results, "StringRef":$callee,
                   CArg<"ValueRange", "{}">:$args)>,
    OpBuilder<(ins "LLVMFunctionType":$calleeType, "StringAttr":$callee,
                   CArg<"ValueRange", "{}">:$args)>,
    OpBuilder<(ins "LLVMFunctionType":$calleeType, "FlatSymbolRefAttr":$callee,
                   CArg<"ValueRange", "{}">:$args)>,
    OpBuilder<(ins "LLVMFunctionType":$calleeType, "StringRef":$callee,
                   CArg<"ValueRange", "{}">:$args)>
  ];
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    /// Returns the callee function type.
    LLVMFunctionType getCalleeFunctionType();
  }];
}

//===----------------------------------------------------------------------===//
// ExtractElementOp
//===----------------------------------------------------------------------===//

def LLVM_ExtractElementOp : LLVM_Op<"extractelement", [Pure,
    TypesMatchWith<"result type matches vector element type", "vector", "res",
                   "LLVM::getVectorElementType($_self)">]> {
  let summary = "Extract an element from an LLVM vector.";

  let arguments = (ins LLVM_AnyVector:$vector, AnySignlessInteger:$position);
  let results = (outs LLVM_Type:$res);

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
                       AnySignlessInteger:$position);
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
          [Pure, AllTypesMatch<["trueValue", "falseValue", "res"]>,
           DeclareOpInterfaceMethods<FastmathFlagsInterface>,
           DeclareOpInterfaceMethods<SelectLikeOpInterface>]>,
      LLVM_Builder<
          "$res = builder.CreateSelect($condition, $trueValue, $falseValue);"> {
  let arguments = (ins LLVM_ScalarOrVectorOf<I1>:$condition,
                   LLVM_Type:$trueValue, LLVM_Type:$falseValue,
                   DefaultValuedAttr<LLVM_FastmathFlagsAttr,
                                     "{}">:$fastmathFlags);
  let results = (outs LLVM_Type:$res);
  let assemblyFormat = "operands attr-dict `:` type($condition) `,` type($res)";
  string llvmInstName = "Select";
  string mlirBuilder = [{
    auto op = $_builder.create<LLVM::SelectOp>(
      $_location, $_resultType, $condition, $trueValue, $falseValue);
    moduleImport.setFastmathFlagsAttr(inst, op);
    $res = op;
  }];
}
def LLVM_FreezeOp : LLVM_Op<"freeze", [Pure, SameOperandsAndResultType]> {
  let arguments = (ins LLVM_Type:$val);
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "$val attr-dict `:` type($val)";
  string llvmInstName = "Freeze";
  string llvmBuilder = "$res = builder.CreateFreeze($val);";
  string mlirBuilder = [{
    $res = $_builder.create<LLVM::FreezeOp>($_location, $val);
  }];
}

// Terminators.
def LLVM_BrOp : LLVM_TerminatorOp<"br",
    [DeclareOpInterfaceMethods<BranchOpInterface>, Pure]> {
  let arguments = (ins
    Variadic<LLVM_Type>:$destOperands,
    OptionalAttr<LoopAnnotationAttr>:$loop_annotation
  );
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
  let builders = [
    OpBuilder<(ins "Block *":$dest), [{
      build($_builder, $_state, ValueRange(), dest);
    }]>,
    OpBuilder<(ins "ValueRange":$operands, "Block *":$dest), [{
      build($_builder, $_state, operands, /*loop_annotation=*/{}, dest);
    }]>,
    LLVM_TerminatorPassthroughOpBuilder
  ];
}
def LLVM_CondBrOp : LLVM_TerminatorOp<"cond_br",
    [AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<BranchOpInterface>,
     DeclareOpInterfaceMethods<BranchWeightOpInterface>,
     Pure]> {
  let arguments = (ins I1:$condition,
                   Variadic<LLVM_Type>:$trueDestOperands,
                   Variadic<LLVM_Type>:$falseDestOperands,
                   OptionalAttr<DenseI32ArrayAttr>:$branch_weights,
                   OptionalAttr<LoopAnnotationAttr>:$loop_annotation);
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
      CArg<"std::optional<std::pair<uint32_t, uint32_t>>", "{}">:$weights)>,
  OpBuilder<(ins "Value":$condition, "Block *":$trueDest,
    "Block *":$falseDest, CArg<"ValueRange", "{}">:$falseOperands),
  [{
      build($_builder, $_state, condition, trueDest, ValueRange(), falseDest,
            falseOperands);
  }]>,
  OpBuilder<(ins "Value":$condition, "ValueRange":$trueOperands, "ValueRange":$falseOperands,
    "DenseI32ArrayAttr":$branchWeights, "Block *":$trueDest, "Block *":$falseDest),
  [{
      build($_builder, $_state, condition, trueOperands, falseOperands, branchWeights,
      {}, trueDest, falseDest);
  }]>, LLVM_TerminatorPassthroughOpBuilder];
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def LLVM_ReturnOp : LLVM_TerminatorOp<"return", [Pure, ReturnLike]> {
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
    FailureOr<SmallVector<Value>> mlirOperands =
      moduleImport.convertValues(llvmOperands);
    if (failed(mlirOperands))
      return failure();
    $_op = $_builder.create<LLVM::ReturnOp>($_location, *mlirOperands);
  }];
}

def LLVM_ResumeOp : LLVM_TerminatorOp<"resume"> {
  let arguments = (ins LLVM_Type:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
  // Consistency of llvm.resume value types is checked in LLVMFuncOp::verify().
  let hasVerifier = false;
  string llvmInstName = "Resume";
  string llvmBuilder = [{ builder.CreateResume($value); }];
  string mlirBuilder = [{
    $_op = $_builder.create<LLVM::ResumeOp>($_location, $value);
  }];
}
def LLVM_UnreachableOp : LLVM_TerminatorOp<"unreachable"> {
  let assemblyFormat = "attr-dict";
  string llvmInstName = "Unreachable";
  string llvmBuilder = [{ builder.CreateUnreachable(); }];
  string mlirBuilder = [{
    $_op = $_builder.create<LLVM::UnreachableOp>($_location);
  }];
}

def LLVM_SwitchOp : LLVM_TerminatorOp<"switch",
    [AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<BranchOpInterface>,
     DeclareOpInterfaceMethods<BranchWeightOpInterface>,
     Pure]> {
  let arguments = (ins
    AnySignlessInteger:$value,
    Variadic<AnyType>:$defaultOperands,
    VariadicOfVariadic<AnyType, "case_operand_segments">:$caseOperands,
    OptionalAttr<AnyIntElementsAttr>:$case_values,
    DenseI32ArrayAttr:$case_operand_segments,
    OptionalAttr<DenseI32ArrayAttr>:$branch_weights
  );
  let successors = (successor
    AnySuccessor:$defaultDestination,
    VariadicSuccessor<AnySuccessor>:$caseDestinations
  );

  let assemblyFormat = [{
    $value `:` type($value) `,`
    $defaultDestination (`(` $defaultOperands^ `:` type($defaultOperands) `)`)?
    custom<SwitchOpCases>(ref(type($value)), $case_values, $caseDestinations,
                                   $caseOperands, type($caseOperands))
    attr-dict
  }];
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$value,
      "Block *":$defaultDestination,
      "ValueRange":$defaultOperands,
      CArg<"ArrayRef<APInt>", "{}">:$caseValues,
      CArg<"BlockRange", "{}">:$caseDestinations,
      CArg<"ArrayRef<ValueRange>", "{}">:$caseOperands,
      CArg<"ArrayRef<int32_t>", "{}">:$branchWeights)>,
    OpBuilder<(ins "Value":$value,
      "Block *":$defaultDestination,
      "ValueRange":$defaultOperands,
      CArg<"ArrayRef<int32_t>", "{}">:$caseValues,
      CArg<"BlockRange", "{}">:$caseDestinations,
      CArg<"ArrayRef<ValueRange>", "{}">:$caseOperands,
      CArg<"ArrayRef<int32_t>", "{}">:$branchWeights)>,
    OpBuilder<(ins "Value":$value,
      "Block *":$defaultDestination,
      "ValueRange":$defaultOperands,
      CArg<"DenseIntElementsAttr", "{}">:$caseValues,
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
    [Pure, ConstantLike, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
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
      %0 = llvm.mlir.addressof @const : !llvm.ptr

      // Use it as a regular pointer.
      %1 = llvm.load %0 : !llvm.ptr -> i32

      // Get the address of a function.
      %2 = llvm.mlir.addressof @foo : !llvm.ptr

      // The function address can be used for indirect calls.
      llvm.call %2() : !llvm.ptr, () -> ()
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
            LLVM::LLVMPointerType::get($_builder.getContext(), global.getAddrSpace()),
            global.getSymName());
      $_state.addAttributes(attrs);
    }]>,
    OpBuilder<(ins "LLVMFuncOp":$func,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      build($_builder, $_state,
            LLVM::LLVMPointerType::get($_builder.getContext()), func.getName());
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

  let hasFolder = 1;
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
    UnitAttr:$externally_initialized,
    OptionalAttr<AnyAttr>:$value,
    OptionalAttr<I64Attr>:$alignment,
    DefaultValuedAttr<ConfinedAttr<I32Attr, [IntNonNegative]>, "0">:$addr_space,
    OptionalAttr<UnnamedAddr>:$unnamed_addr,
    OptionalAttr<StrAttr>:$section,
    OptionalAttr<SymbolRefAttr>:$comdat,
    OptionalAttr<DIGlobalVariableExpressionArrayAttr>:$dbg_exprs,
    DefaultValuedAttr<Visibility, "mlir::LLVM::Visibility::Default">:$visibility_
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
    llvm.mlir.global constant @int_gep() : !llvm.ptr {
      %0 = llvm.mlir.addressof @g2 : !llvm.ptr
      %1 = llvm.mlir.constant(2 : i32) : i32
      %2 = llvm.getelementptr %0[%1]
         : (!llvm.ptr, i32) -> !llvm.ptr, i32
      // The initializer region must end with `llvm.return`.
      llvm.return %2 : !llvm.ptr
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
    llvm.mlir.global constant @int_gep() : !llvm.ptr {
      %0 = llvm.mlir.addressof @g2 : !llvm.ptr
      %1 = llvm.mlir.constant(2 : i32) : i32
      %2 = llvm.getelementptr %0[%1]
         : (!llvm.ptr, i32) -> !llvm.ptr, i32
      llvm.return %2 : !llvm.ptr
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
      CArg<"SymbolRefAttr", "{}">:$comdat,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<Attribute>", "{}">:$dbgExprs)>
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

def LLVM_ComdatSelectorOp : LLVM_Op<"comdat_selector", [Symbol]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    Comdat:$comdat
  );

  let summary = "LLVM dialect comdat selector declaration";

  let description = [{
    Provides access to object file COMDAT section/group functionality.

    Examples:
    ```mlir
    llvm.comdat @__llvm_comdat {
      llvm.comdat_selector @any any
    }
    llvm.mlir.global internal constant @has_any_comdat(1 : i64) comdat(@__llvm_comdat::@any) : i64
    ```
  }];
  let assemblyFormat = "$sym_name $comdat attr-dict";
}

def LLVM_ComdatOp : LLVM_Op<"comdat", [NoTerminator, NoRegionArguments, SymbolTable, Symbol]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name
  );
  let summary = "LLVM dialect comdat region";

  let description = [{
    Provides access to object file COMDAT section/group functionality.

    Examples:
    ```mlir
    llvm.comdat @__llvm_comdat {
      llvm.comdat_selector @any any
    }
    llvm.mlir.global internal constant @has_any_comdat(1 : i64) comdat(@__llvm_comdat::@any) : i64
    ```
  }];
  let regions = (region SizedRegion<1>:$body);


  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "StringRef":$symName)>];

  let assemblyFormat = "$sym_name $body attr-dict";
  let hasRegionVerifier = 1;
}

def LLVM_LLVMFuncOp : LLVM_Op<"func", [
    AutomaticAllocationScope, IsolatedFromAbove, FunctionOpInterface
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
    OptionalAttr<StrAttr>:$sym_visibility,
    TypeAttrOf<LLVM_FunctionType>:$function_type,
    DefaultValuedAttr<Linkage, "Linkage::External">:$linkage,
    UnitAttr:$dso_local,
    DefaultValuedAttr<CConv, "CConv::C">:$CConv,
    OptionalAttr<SymbolRefAttr>:$comdat,
    OptionalAttr<UnitAttr>:$convergent,
    OptionalAttr<FlatSymbolRefAttr>:$personality,
    OptionalAttr<StrAttr>:$garbageCollector,
    OptionalAttr<ArrayAttr>:$passthrough,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    OptionalAttr<I64Attr>:$function_entry_count,
    OptionalAttr<LLVM_MemoryEffectsAttr>:$memory_effects,
    DefaultValuedAttr<Visibility, "mlir::LLVM::Visibility::Default">:$visibility_,
    OptionalAttr<UnitAttr>:$arm_streaming,
    OptionalAttr<UnitAttr>:$arm_locally_streaming,
    OptionalAttr<UnitAttr>:$arm_streaming_compatible,
    OptionalAttr<UnitAttr>:$arm_new_za,
    OptionalAttr<UnitAttr>:$arm_in_za,
    OptionalAttr<UnitAttr>:$arm_out_za,
    OptionalAttr<UnitAttr>:$arm_inout_za,
    OptionalAttr<UnitAttr>:$arm_preserves_za,
    OptionalAttr<StrAttr>:$section,
    OptionalAttr<UnnamedAddr>:$unnamed_addr,
    OptionalAttr<I64Attr>:$alignment,
    OptionalAttr<LLVM_VScaleRangeAttr>:$vscale_range,
    OptionalAttr<FramePointerKindAttr>:$frame_pointer,
    OptionalAttr<StrAttr>:$target_cpu,
    OptionalAttr<StrAttr>:$tune_cpu,
    OptionalAttr<LLVM_TargetFeaturesAttr>:$target_features,
    OptionalAttr<BoolAttr>:$unsafe_fp_math,
    OptionalAttr<BoolAttr>:$no_infs_fp_math,
    OptionalAttr<BoolAttr>:$no_nans_fp_math,
    OptionalAttr<BoolAttr>:$approx_func_fp_math,
    OptionalAttr<BoolAttr>:$no_signed_zeros_fp_math,
    OptionalAttr<StrAttr>:$denormal_fp_math,
    OptionalAttr<StrAttr>:$denormal_fp_math_f32,
    OptionalAttr<StrAttr>:$fp_contract,
    OptionalAttr<UnitAttr>:$no_inline,
    OptionalAttr<UnitAttr>:$always_inline,
    OptionalAttr<UnitAttr>:$no_unwind,
    OptionalAttr<UnitAttr>:$will_return,
    OptionalAttr<UnitAttr>:$optimize_none,
    OptionalAttr<LLVM_VecTypeHintAttr>:$vec_type_hint,
    OptionalAttr<DenseI32ArrayAttr>:$work_group_size_hint,
    OptionalAttr<DenseI32ArrayAttr>:$reqd_work_group_size,
    OptionalAttr<I32Attr>:$intel_reqd_sub_group_size
  );

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "StringRef":$name, "Type":$type,
      CArg<"Linkage", "Linkage::External">:$linkage,
      CArg<"bool", "false">:$dsoLocal,
      CArg<"CConv", "CConv::C">:$cconv,
      CArg<"SymbolRefAttr", "{}">:$comdat,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs,
      CArg<"std::optional<uint64_t>", "{}">:$functionEntryCount)>
  ];

  let extraClassDeclaration = [{
    // Add an entry block to an empty function, and set up the block arguments
    // to match the signature of the function.
    Block *addEntryBlock(OpBuilder &builder);

    bool isVarArg() { return getFunctionType().isVarArg(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getParams(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() {
      if (::llvm::isa<LLVM::LLVMVoidType>(getFunctionType().getReturnType()))
        return {};
      return getFunctionType().getReturnTypes();
    }

    /// Returns the callable region, which is the function body. If the function
    /// is external, returns null.
    Region *getCallableRegion();

    /// Returns true if the `no_inline` attribute is set, false otherwise.
    bool isNoInline() { return bool(getNoInlineAttr()); }

    /// Returns true if the `always_inline` attribute is set, false otherwise.
    bool isAlwaysInline() { return bool(getAlwaysInlineAttr()); }

    /// Returns true if the `optimize_none` attribute is set, false otherwise.
    bool isOptimizeNone() { return bool(getOptimizeNoneAttr()); }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def LLVM_NoneTokenOp
    : LLVM_Op<"mlir.none", [Pure]> {
  let summary = "Defines a value containing an empty token to LLVM type.";
  let description = [{
    Unlike LLVM IR, MLIR does not have first-class token values. They must be
    explicitly created as SSA values using `llvm.mlir.none`. This operation has
    no operands or attributes, and returns a none token value of a wrapped LLVM IR
    pointer type.

    Examples:

    ```mlir
    %0 = llvm.mlir.none : !llvm.token
    ```
  }];

  string llvmBuilder = [{
    $res = llvm::ConstantTokenNone::get(builder.getContext());
  }];

  let results = (outs LLVM_TokenType:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "attr-dict `:` type($res)";
}

def LLVM_UndefOp : LLVM_Op<"mlir.undef", [Pure, ConstantLike]>,
                   LLVM_Builder<"$res = llvm::UndefValue::get($_resultType);"> {
  let summary = "Creates an undefined value of LLVM dialect type.";
  let description = [{
    Unlike LLVM IR, MLIR does not have first-class undefined values. Such values
    must be created as SSA values using `llvm.mlir.undef`. This operation has no
    operands or attributes. It creates an undefined value of the specified LLVM
    IR dialect type.

    Example:

    ```mlir
    // Create a structure with a 32-bit integer followed by a float.
    %0 = llvm.mlir.undef : !llvm.struct<(i32, f32)>
    ```
  }];
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "attr-dict `:` type($res)";
  let hasFolder = 1;
}

def LLVM_PoisonOp : LLVM_Op<"mlir.poison", [Pure, ConstantLike]>,
                    LLVM_Builder<"$res = llvm::PoisonValue::get($_resultType);"> {
  let summary = "Creates a poison value of LLVM dialect type.";
  let description = [{
    Unlike LLVM IR, MLIR does not have first-class poison values. Such values
    must be created as SSA values using `llvm.mlir.poison`. This operation has
    no operands or attributes. It creates a poison value of the specified LLVM
    IR dialect type.

    Example:

    ```mlir
    // Create a poison value for a structure with a 32-bit integer followed
    // by a float.
    %0 = llvm.mlir.poison : !llvm.struct<(i32, f32)>
    ```
  }];
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "attr-dict `:` type($res)";
  let hasFolder = 1;
}

def LLVM_ZeroOp
    : LLVM_Op<"mlir.zero", [Pure, ConstantLike]>,
      LLVM_Builder<"$res = llvm::Constant::getNullValue($_resultType);">
{
  let summary = "Creates a zero-initialized value of LLVM dialect type.";
  let description = [{
    Unlike LLVM IR, MLIR does not have first-class zero-initialized values.
    Such values must be created as SSA values using `llvm.mlir.zero`. This
    operation has no operands or attributes. It creates a zero-initialized
    value of the specified LLVM IR dialect type.

    Example:

    ```mlir
    // Create a zero-initialized value for a structure with a 32-bit integer
    // followed by a float.
    %0 = llvm.mlir.zero : !llvm.struct<(i32, f32)>
    ```
  }];
  let results = (outs LLVM_Type:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "attr-dict `:` type($res)";
  let hasVerifier = 1;
  let hasFolder = 1;
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
    operations. `llvm.mlir.constant` creates such values for scalars, vectors,
    strings, and structs. It has a mandatory `value` attribute whose type
    depends on the type of the constant value. The type of the constant value
    must correspond to the attribute type converted to LLVM IR type.

    When creating constant scalars, the `value` attribute must be either an
    integer attribute or a floating point attribute. The type of the attribute
    may be omitted for `i64` and `f64` types that are implied.

    When creating constant vectors, the `value` attribute must be either an
    array attribute, a dense attribute, or a sparse attribute that contains
    integers or floats. The number of elements in the result vector must match
    the number of elements in the attribute.

    When creating constant strings, the `value` attribute must be a string
    attribute. The type of the constant must be an LLVM array of `i8`s, and the
    length of the array must match the length of the attribute.

    When creating constant structs, the `value` attribute must be an array
    attribute that contains integers or floats. The type of the constant must be
    an LLVM struct type. The number of fields in the struct must match the
    number of elements in the attribute, and the type of each LLVM struct field
    must correspond to the type of the corresponding attribute element converted
    to LLVM IR.

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

  let extraClassDeclaration = [{
    /// Whether the constant op can be constructed with a particular value and
    /// type.
    static bool isBuildableWith(Attribute value, Type type);

    /// Build the constant op with `value` and `type` if possible, otherwise
    /// returns null.
    static ConstantOp materialize(OpBuilder &builder, Attribute value,
                                  Type type, Location loc);
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
}

// Atomic operations.
//

def LLVM_AtomicRMWType
    : AnyTypeOf<[LLVM_AnyFloat, LLVM_AnyPointer, AnySignlessInteger, LLVM_AnyFixedVector]>;

def LLVM_AtomicRMWOp : LLVM_MemAccessOpBase<"atomicrmw", [
      TypesMatchWith<"result #0 and operand #1 have the same type",
                     "val", "res", "$_self">]> {
  dag args = (ins AtomicBinOp:$bin_op,
              LLVM_AnyPointer:$ptr,
              LLVM_AtomicRMWType:$val, AtomicOrdering:$ordering,
              OptionalAttr<StrAttr>:$syncscope,
              OptionalAttr<I64Attr>:$alignment,
              UnitAttr:$volatile_);
  // Append the aliasing related attributes defined in LLVM_MemAccessOpBase.
  let arguments = !con(args, aliasAttrs);
  let results = (outs LLVM_AtomicRMWType:$res);
  let assemblyFormat = [{
    (`volatile` $volatile_^)? $bin_op $ptr `,` $val
    (`syncscope` `(` $syncscope^ `)`)? $ordering attr-dict `:`
    qualified(type($ptr)) `,` type($val)
  }];
  string llvmInstName = "AtomicRMW";
  string llvmBuilder = [{
    auto *inst = builder.CreateAtomicRMW(
        convertAtomicBinOpToLLVM($bin_op), $ptr, $val, llvm::MaybeAlign(),
        convertAtomicOrderingToLLVM($ordering));
    $res = inst;
  }] # setVolatileCode
     # setSyncScopeCode
     # setAlignmentCode
     # setAccessGroupsMetadataCode
     # setAliasAnalysisMetadataCode;
  string mlirBuilder = [{
    auto *atomicInst = cast<llvm::AtomicRMWInst>(inst);
    unsigned alignment = atomicInst->getAlign().value();
    $res = $_builder.create<LLVM::AtomicRMWOp>($_location,
        convertAtomicBinOpFromLLVM(atomicInst->getOperation()), $ptr, $val,
        convertAtomicOrderingFromLLVM(atomicInst->getOrdering()),
        getLLVMSyncScope(atomicInst), alignment, atomicInst->isVolatile());
  }];
  list<int> llvmArgIndices = [-1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1];
  let builders = [
    OpBuilder<(ins "LLVM::AtomicBinOp":$binOp, "Value":$ptr, "Value":$val,
      "LLVM::AtomicOrdering":$ordering,
      CArg<"StringRef", "StringRef()">:$syncscope,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isVolatile
    )>
  ];
  let hasVerifier = 1;
}

def LLVM_AtomicCmpXchgType : AnyTypeOf<[AnySignlessInteger, LLVM_AnyPointer]>;

def LLVM_AtomicCmpXchgOp : LLVM_MemAccessOpBase<"cmpxchg", [
      TypesMatchWith<"operand #1 and operand #2 have the same type",
                     "val", "cmp", "$_self">,
      TypesMatchWith<"result #0 has an LLVM struct type consisting of "
                     "the type of operand #2 and a bool", "val", "res",
                     "getValAndBoolStructType($_self)">]> {
  dag args = (ins LLVM_AnyPointer:$ptr,
              LLVM_AtomicCmpXchgType:$cmp, LLVM_AtomicCmpXchgType:$val,
              AtomicOrdering:$success_ordering,
              AtomicOrdering:$failure_ordering,
              OptionalAttr<StrAttr>:$syncscope,
              OptionalAttr<I64Attr>:$alignment,
              UnitAttr:$weak,
              UnitAttr:$volatile_);
  // Append the aliasing related attributes defined in LLVM_MemAccessOpBase.
  let arguments = !con(args, aliasAttrs);
  let results = (outs LLVM_AnyStruct:$res);
  let assemblyFormat = [{
    (`weak` $weak^)? (`volatile` $volatile_^)? $ptr `,` $cmp `,` $val
    (`syncscope` `(` $syncscope^ `)`)? $success_ordering $failure_ordering
    attr-dict `:` qualified(type($ptr)) `,` type($val)
  }];
  string llvmInstName = "AtomicCmpXchg";
  string llvmBuilder = [{
    auto *inst = builder.CreateAtomicCmpXchg($ptr, $cmp, $val,
        llvm::MaybeAlign(), convertAtomicOrderingToLLVM($success_ordering),
        convertAtomicOrderingToLLVM($failure_ordering));
    $res = inst;
    inst->setWeak($weak);
  }] # setVolatileCode
     # setSyncScopeCode
     # setAlignmentCode
     # setAccessGroupsMetadataCode
     # setAliasAnalysisMetadataCode;
  string mlirBuilder = [{
    auto *cmpXchgInst = cast<llvm::AtomicCmpXchgInst>(inst);
    unsigned alignment = cmpXchgInst->getAlign().value();
    $res = $_builder.create<LLVM::AtomicCmpXchgOp>(
      $_location, $ptr, $cmp, $val,
      convertAtomicOrderingFromLLVM(cmpXchgInst->getSuccessOrdering()),
      convertAtomicOrderingFromLLVM(cmpXchgInst->getFailureOrdering()),
      getLLVMSyncScope(cmpXchgInst), alignment, cmpXchgInst->isWeak(),
      cmpXchgInst->isVolatile());
  }];
  let builders = [
    OpBuilder<(ins "Value":$ptr, "Value":$cmp, "Value":$val,
      "LLVM::AtomicOrdering":$successOrdering,
      "LLVM::AtomicOrdering":$failureOrdering,
      CArg<"StringRef", "StringRef()">:$syncscope,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isWeak,
      CArg<"bool", "false">:$isVolatile
    )>
  ];
  let hasVerifier = 1;
}

def LLVM_FenceOp : LLVM_Op<"fence">, LLVM_MemOpPatterns {
  let arguments = (ins AtomicOrdering:$ordering,
                   OptionalAttr<StrAttr>:$syncscope);
  let assemblyFormat = "(`syncscope` `(` $syncscope^ `)`)? $ordering attr-dict";
  string llvmInstName = "Fence";
  let llvmBuilder = [{
    auto *inst = builder.CreateFence(convertAtomicOrderingToLLVM($ordering));
  }] # setSyncScopeCode;
  string mlirBuilder = [{
    llvm::FenceInst *fenceInst = cast<llvm::FenceInst>(inst);
    $_op = $_builder.create<LLVM::FenceOp>(
      $_location,
      convertAtomicOrderingFromLLVM(fenceInst->getOrdering()),
      getLLVMSyncScope(fenceInst));
  }];
  let builders = [
    LLVM_VoidResultTypeOpBuilder,
    LLVM_ZeroResultOpBuilder,
    OpBuilder<(ins "LLVM::AtomicOrdering":$ordering,
      CArg<"StringRef", "StringRef()">:$syncscope)>
  ];
  let hasVerifier = 1;
}

def LLVM_InlineAsmOp : LLVM_Op<"inline_asm", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
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

//===--------------------------------------------------------------------===//
// CallIntrinsicOp
//===--------------------------------------------------------------------===//

def LLVM_CallIntrinsicOp
    : LLVM_Op<"call_intrinsic",
              [AttrSizedOperandSegments,
               DeclareOpInterfaceMethods<FastmathFlagsInterface>]> {
  let summary = "Call to an LLVM intrinsic function.";
  let description = [{
    Call the specified llvm intrinsic. If the intrinsic is overloaded, use
    the MLIR function type of this op to determine which intrinsic to call.
  }];
  let arguments = (ins StrAttr:$intrin, Variadic<LLVM_Type>:$args,
                       DefaultValuedAttr<LLVM_FastmathFlagsAttr,
                                         "{}">:$fastmathFlags,
                       VariadicOfVariadic<LLVM_Type,
                                          "op_bundle_sizes">:$op_bundle_operands,
                       DenseI32ArrayAttr:$op_bundle_sizes,
                       OptionalAttr<ArrayAttr>:$op_bundle_tags);
  let results = (outs Optional<LLVM_Type>:$results);
  let llvmBuilder = [{
    return convertCallLLVMIntrinsicOp(op, builder, moduleTranslation);
  }];
  let assemblyFormat = [{
    $intrin `(` $args `)`
    ( custom<OpBundles>($op_bundle_operands, type($op_bundle_operands),
                        $op_bundle_tags)^ )?
    `:` functional-type($args, $results)
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins "StringAttr":$intrin, "ValueRange":$args)>,
    OpBuilder<(ins "StringAttr":$intrin, "ValueRange":$args, "FastmathFlagsAttr":$fastMathFlags)>,
    OpBuilder<(ins "Type": $resultType, "StringAttr":$intrin, "ValueRange":$args)>,
    OpBuilder<(ins "TypeRange": $resultTypes, "StringAttr":$intrin, "ValueRange":$args, "FastmathFlagsAttr":$fastMathFlags)>
  ];

  let hasVerifier = 1;
}

def LLVM_LinkerOptionsOp
    : LLVM_Op<"linker_options"> {
  let summary = "Options to pass to the linker when the object file is linked";
  let description = [{
    Pass the given options to the linker when the resulting object file is linked.
    This is used extensively on Windows to determine the C runtime that the object
    files should link against.

    Examples:
    ```mlir
    // Link against the MSVC static threaded CRT.
    llvm.linker_options ["/DEFAULTLIB:", "libcmt"]

    // Link against aarch64 compiler-rt builtins
    llvm.linker_options ["-l", "clang_rt.builtins-aarch64"]
    ```
  }];
  let arguments  = (ins StrArrayAttr:$options);
  let assemblyFormat = [{
    $options attr-dict
  }];

  let llvmBuilder = [{
    convertLinkerOptionsOp($options, builder, moduleTranslation);
  }];

  let hasVerifier = 1;
}

#endif // LLVMIR_OPS
