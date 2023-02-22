#ifndef LLVM_INTRINSIC_OPS
#define LLVM_INTRINSIC_OPS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMAttrDefs.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/LLVMIR/LLVMAttrDefs.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// Operations that correspond to LLVM intrinsics. With MLIR operation set being
// extendable, there is no reason to introduce a hard boundary between "core"
// operations and intrinsics. However, we systematically prefix them with
// "intr." to avoid potential name clashes.

class LLVM_UnaryIntrOpBase<string func, Type element,
                           list<Trait> traits = [], bit requiresFastmath = 0> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([Pure, SameOperandsAndResultType], traits),
           requiresFastmath> {
  dag commonArgs = (ins LLVM_ScalarOrVectorOf<element>:$in);
  let assemblyFormat = "`(` operands `)` custom<LLVMOpAttrs>(attr-dict) `:` "
      "functional-type(operands, results)";
}

class LLVM_UnaryIntrOpI<string func, list<Trait> traits = []> :
    LLVM_UnaryIntrOpBase<func, AnySignlessInteger, traits> {
  let arguments = commonArgs;
}

class LLVM_UnaryIntrOpF<string func, list<Trait> traits = []> :
    LLVM_UnaryIntrOpBase<func, LLVM_AnyFloat, traits, /*requiresFastmath=*/1> {
  dag fmfArg = (
    ins DefaultValuedAttr<LLVM_FastmathFlagsAttr, "{}">:$fastmathFlags);
  let arguments = !con(commonArgs, fmfArg);
}

class LLVM_BinarySameArgsIntrOpBase<string func, Type element,
              list<Trait> traits = [], bit requiresFastmath = 0> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([Pure, SameOperandsAndResultType], traits),
           requiresFastmath> {
  dag commonArgs = (ins LLVM_ScalarOrVectorOf<element>:$a,
                        LLVM_ScalarOrVectorOf<element>:$b);
  let assemblyFormat = "`(` operands `)` custom<LLVMOpAttrs>(attr-dict) `:` "
      "functional-type(operands, results)";
}

class LLVM_BinarySameArgsIntrOpI<string func, list<Trait> traits = []> :
    LLVM_BinarySameArgsIntrOpBase<func, AnySignlessInteger, traits> {
  let arguments = commonArgs;
}

class LLVM_BinarySameArgsIntrOpF<string func, list<Trait> traits = []> :
    LLVM_BinarySameArgsIntrOpBase<func, LLVM_AnyFloat, traits,
                                  /*requiresFastmath=*/1> {
  dag fmfArg = (
    ins DefaultValuedAttr<LLVM_FastmathFlagsAttr, "{}">:$fastmathFlags);
  let arguments = !con(commonArgs, fmfArg);
}

class LLVM_TernarySameArgsIntrOpF<string func, list<Trait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([Pure, SameOperandsAndResultType], traits),
           /*requiresFastmath=*/1> {
  let arguments = (ins LLVM_ScalarOrVectorOf<AnyFloat>:$a,
                       LLVM_ScalarOrVectorOf<AnyFloat>:$b,
                       LLVM_ScalarOrVectorOf<AnyFloat>:$c,
                       DefaultValuedAttr<LLVM_FastmathFlagsAttr,
                                         "{}">:$fastmathFlags);
  let assemblyFormat = "`(` operands `)` custom<LLVMOpAttrs>(attr-dict) `:` "
      "functional-type(operands, results)";
}

class LLVM_CountZerosIntrOp<string func, list<Trait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([Pure], traits)> {
  let arguments = (ins LLVM_ScalarOrVectorOf<AnySignlessInteger>:$in,
                   I1:$zero_undefined);
}

def LLVM_AbsOp : LLVM_OneResultIntrOp<"abs", [], [0], [Pure]> {
  let arguments = (ins LLVM_ScalarOrVectorOf<AnySignlessInteger>:$in,
                   I1:$is_int_min_poison);
}

def LLVM_IsFPClass : LLVM_OneResultIntrOp<"is.fpclass", [], [0], [Pure]> {
  let arguments = (ins LLVM_ScalarOrVectorOf<LLVM_AnyFloat>:$in, I32:$bit);
}

def LLVM_CopySignOp : LLVM_BinarySameArgsIntrOpF<"copysign">;
def LLVM_CosOp : LLVM_UnaryIntrOpF<"cos">;
def LLVM_ExpOp : LLVM_UnaryIntrOpF<"exp">;
def LLVM_Exp2Op : LLVM_UnaryIntrOpF<"exp2">;
def LLVM_FAbsOp : LLVM_UnaryIntrOpF<"fabs">;
def LLVM_FCeilOp : LLVM_UnaryIntrOpF<"ceil">;
def LLVM_FFloorOp : LLVM_UnaryIntrOpF<"floor">;
def LLVM_FMAOp : LLVM_TernarySameArgsIntrOpF<"fma">;
def LLVM_FMulAddOp : LLVM_TernarySameArgsIntrOpF<"fmuladd">;
def LLVM_Log10Op : LLVM_UnaryIntrOpF<"log10">;
def LLVM_Log2Op : LLVM_UnaryIntrOpF<"log2">;
def LLVM_LogOp : LLVM_UnaryIntrOpF<"log">;
def LLVM_Prefetch : LLVM_ZeroResultIntrOp<"prefetch", [0]> {
  let arguments = (ins LLVM_AnyPointer:$addr, I32:$rw, I32:$hint, I32:$cache);
}
def LLVM_SinOp : LLVM_UnaryIntrOpF<"sin">;
def LLVM_RoundEvenOp : LLVM_UnaryIntrOpF<"roundeven">;
def LLVM_RoundOp : LLVM_UnaryIntrOpF<"round">;
def LLVM_FTruncOp : LLVM_UnaryIntrOpF<"trunc">;
def LLVM_SqrtOp : LLVM_UnaryIntrOpF<"sqrt">;
def LLVM_PowOp : LLVM_BinarySameArgsIntrOpF<"pow">;
def LLVM_PowIOp : LLVM_OneResultIntrOp<"powi", [], [0,1],
    [DeclareOpInterfaceMethods<FastmathFlagsInterface>, Pure],
    /*requiresFastmath=*/1> {
  let arguments =
      (ins LLVM_ScalarOrVectorOf<LLVM_AnyFloat>:$val,
           AnySignlessInteger:$power,
           DefaultValuedAttr<LLVM_FastmathFlagsAttr, "{}">:$fastmathFlags);
  let assemblyFormat = "`(` operands `)` custom<LLVMOpAttrs>(attr-dict) `:` "
      "functional-type(operands, results)";
}
def LLVM_BitReverseOp : LLVM_UnaryIntrOpI<"bitreverse">;
def LLVM_CountLeadingZerosOp : LLVM_CountZerosIntrOp<"ctlz">;
def LLVM_CountTrailingZerosOp : LLVM_CountZerosIntrOp<"cttz">;
def LLVM_CtPopOp : LLVM_UnaryIntrOpI<"ctpop">;
def LLVM_MaxNumOp : LLVM_BinarySameArgsIntrOpF<"maxnum">;
def LLVM_MinNumOp : LLVM_BinarySameArgsIntrOpF<"minnum">;
def LLVM_MaximumOp : LLVM_BinarySameArgsIntrOpF<"maximum">;
def LLVM_MinimumOp : LLVM_BinarySameArgsIntrOpF<"minimum">;
def LLVM_SMaxOp : LLVM_BinarySameArgsIntrOpI<"smax">;
def LLVM_SMinOp : LLVM_BinarySameArgsIntrOpI<"smin">;
def LLVM_UMaxOp : LLVM_BinarySameArgsIntrOpI<"umax">;
def LLVM_UMinOp : LLVM_BinarySameArgsIntrOpI<"umin">;

def LLVM_MemcpyOp : LLVM_ZeroResultIntrOp<"memcpy", [0, 1, 2]> {
  let arguments = (ins Arg<LLVM_AnyPointer,"",[MemWrite]>:$dst,
                       Arg<LLVM_AnyPointer,"",[MemRead]>:$src,
                       AnySignlessInteger:$len, I1:$isVolatile);
}
def LLVM_MemcpyInlineOp : LLVM_ZeroResultIntrOp<"memcpy.inline", [0, 1, 2]> {
  let arguments = (ins Arg<LLVM_AnyPointer,"",[MemWrite]>:$dst,
                       Arg<LLVM_AnyPointer,"",[MemRead]>:$src,
                       AnySignlessInteger:$len, I1:$isVolatile);
}
def LLVM_MemmoveOp : LLVM_ZeroResultIntrOp<"memmove", [0, 1, 2]> {
  let arguments = (ins Arg<LLVM_AnyPointer,"",[MemWrite]>:$dst,
                       Arg<LLVM_AnyPointer,"",[MemRead]>:$src,
                       AnySignlessInteger:$len, I1:$isVolatile);
}

def LLVM_MemsetOp : LLVM_ZeroResultIntrOp<"memset", [0, 2]> {
  let arguments = (ins Arg<LLVM_AnyPointer,"",[MemWrite]>:$dst,
                       I8:$val, AnySignlessInteger:$len, I1:$isVolatile);
}

//===----------------------------------------------------------------------===//
// Lifetime Markers
//===----------------------------------------------------------------------===//

/// Base operation for lifetime markers. The LLVM intrinsics require the size
/// operand to be an immediate. In MLIR it is encoded as an attribute.
class LLVM_LifetimeBaseOp<string opName> : LLVM_ZeroResultIntrOp<opName> {
  let arguments = (ins I64Attr:$size, LLVM_AnyPointer:$ptr);

  // Custom builder to convert the size attribute to an integer.
  let llvmBuilder = [{
    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module, llvm::Intrinsic::}] # llvmEnumName # [{, {}] #
        !interleave(ListIntSubst<LLVM_IntrPatterns.operand, [0]>.lst, ", ")
        # [{});
    builder.CreateCall(fn, {builder.getInt64(op.getSizeAttr().getInt()),
                            moduleTranslation.lookupValue(op.getPtr())});
  }];

  let assemblyFormat = "$size `,` $ptr attr-dict `:` qualified(type($ptr))";
}

def LLVM_LifetimeStartOp : LLVM_LifetimeBaseOp<"lifetime.start"> {
  // Custom builder to convert the size argument to an attribute.
  string mlirBuilder = [{
    $_op = $_builder.create<LLVM::LifetimeStartOp>(
      $_location, $_int_attr($size), $ptr);
  }];
}
def LLVM_LifetimeEndOp : LLVM_LifetimeBaseOp<"lifetime.end"> {
  // Custom builder to convert the size argument to an attribute.
  string mlirBuilder = [{
    $_op = $_builder.create<LLVM::LifetimeEndOp>(
      $_location, $_int_attr($size), $ptr);
  }];
}

// Intrinsics with multiple returns.

class LLVM_ArithWithOverflowOp<string mnem>
    : LLVM_IntrOp<mnem, [0], [], [SameOperandsElementType], 2>,
      Arguments<(ins LLVM_ScalarOrVectorOf<AnySignlessInteger>,
                 LLVM_ScalarOrVectorOf<AnySignlessInteger>)>;

def LLVM_SAddWithOverflowOp : LLVM_ArithWithOverflowOp<"sadd.with.overflow">;
def LLVM_UAddWithOverflowOp : LLVM_ArithWithOverflowOp<"uadd.with.overflow">;
def LLVM_SSubWithOverflowOp : LLVM_ArithWithOverflowOp<"ssub.with.overflow">;
def LLVM_USubWithOverflowOp : LLVM_ArithWithOverflowOp<"usub.with.overflow">;
def LLVM_SMulWithOverflowOp : LLVM_ArithWithOverflowOp<"smul.with.overflow">;
def LLVM_UMulWithOverflowOp : LLVM_ArithWithOverflowOp<"umul.with.overflow">;

def LLVM_AssumeOp
  : LLVM_ZeroResultIntrOp<"assume", []>, Arguments<(ins I1:$cond)>;

//
// Coroutine intrinsics.
//

def LLVM_CoroIdOp : LLVM_IntrOp<"coro.id", [], [], [], 1> {
  let arguments = (ins I32:$align,
                       LLVM_i8Ptr:$promise,
                       LLVM_i8Ptr:$coroaddr,
                       LLVM_i8Ptr:$fnaddrs);
  let assemblyFormat = "$align `,` $promise `,` $coroaddr `,` $fnaddrs"
    " attr-dict `:` functional-type(operands, results)";
}

def LLVM_CoroBeginOp : LLVM_IntrOp<"coro.begin", [], [], [], 1> {
  let arguments = (ins LLVM_TokenType:$token,
                       LLVM_i8Ptr:$mem);
  let assemblyFormat = "$token `,` $mem attr-dict `:` functional-type(operands, results)";
}

def LLVM_CoroSizeOp : LLVM_IntrOp<"coro.size", [0], [], [], 1> {
  let assemblyFormat = "attr-dict `:` type($res)";
}

def LLVM_CoroAlignOp : LLVM_IntrOp<"coro.align", [0], [], [], 1> {
  let assemblyFormat = "attr-dict `:` type($res)";
}

def LLVM_CoroSaveOp : LLVM_IntrOp<"coro.save", [], [], [], 1> {
  let arguments = (ins LLVM_i8Ptr:$handle);
  let assemblyFormat = "$handle attr-dict `:` functional-type(operands, results)";
}

def LLVM_CoroSuspendOp : LLVM_IntrOp<"coro.suspend", [], [], [], 1> {
  let arguments = (ins LLVM_TokenType:$save,
                       I1:$final);
  let assemblyFormat = "$save `,` $final attr-dict `:` type($res)";
}

def LLVM_CoroEndOp : LLVM_IntrOp<"coro.end", [], [], [], 1> {
  let arguments = (ins LLVM_i8Ptr:$handle,
                       I1:$unwind);
  let assemblyFormat = "$handle `,` $unwind attr-dict `:` functional-type(operands, results)";
}

def LLVM_CoroFreeOp : LLVM_IntrOp<"coro.free", [], [], [], 1> {
  let arguments = (ins LLVM_TokenType:$id,
                       LLVM_i8Ptr:$handle);
  let assemblyFormat = "$id `,` $handle attr-dict `:` functional-type(operands, results)";
}

def LLVM_CoroResumeOp : LLVM_IntrOp<"coro.resume", [], [], [], 0> {
  let arguments = (ins LLVM_i8Ptr:$handle);
  let assemblyFormat = "$handle attr-dict `:` qualified(type($handle))";
}

//
// Debug function intrinsics.
//

class LLVM_DbgIntrOp<string name, string argName> : LLVM_IntrOp<name, [], [], [], 0> {
  let llvmBuilder = [{
    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::LLVMContext &ctx = module->getContext();
    llvm::Function *fn =
      llvm::Intrinsic::getDeclaration(module, llvm::Intrinsic::}]
       # !subst(".", "_", name) # [{);
    builder.CreateCall(fn, {
        llvm::MetadataAsValue::get(ctx,
            llvm::ValueAsMetadata::get(moduleTranslation.lookupValue(opInst.getOperand(0)))),
        llvm::MetadataAsValue::get(ctx, moduleTranslation.translateDebugInfo($varInfo)),
        llvm::MetadataAsValue::get(ctx, llvm::DIExpression::get(ctx, std::nullopt)),
      });
  }];
  let mlirBuilder = [{
    // Drop debug intrinsics with a non-empty debug expression.
    // TODO: Support debug intrinsics that evaluate a debug expression.
    auto *dbgIntr = cast<llvm::DbgVariableIntrinsic>(inst);
    if (dbgIntr->hasArgList() || dbgIntr->getExpression()->getNumElements() != 0)
      return success();
    // Convert the value/address operand late since it cannot be a debug
    // metadata argument list at this stage. Generating the conversion using an
    // argument variable would not work here, since the builder variables are
    // converted before entering the builder, which would result in an error
    // when attempting to convert an argument list.

    FailureOr<Value> argOperand = moduleImport.convertMetadataValue(llvmOperands[0]);
    // Drop the intrinsic when its operand could not be converted. This can
    // happen for use before definition cases that are allowed for debug
    // intrinsics.
    if (failed(argOperand))
      return success();
    $_op = $_builder.create<$_qualCppClassName>($_location,
        *argOperand, $_var_attr($varInfo));
  }];
  let assemblyFormat = [{
    qualified($varInfo) `=` $}] # argName #
      [{ `:` qualified(type($}] # argName # [{)) attr-dict
  }];
}

def LLVM_DbgAddrOp : LLVM_DbgIntrOp<"dbg.addr", "addr"> {
  let summary = "Describe the current address of a local debug info variable.";
  let arguments = (ins LLVM_AnyPointer:$addr, LLVM_DILocalVariableAttr:$varInfo);
}

def LLVM_DbgDeclareOp : LLVM_DbgIntrOp<"dbg.declare", "addr"> {
  let summary = "Declare the address of a local debug info variable.";
  let arguments = (ins LLVM_AnyPointer:$addr, LLVM_DILocalVariableAttr:$varInfo);
}

def LLVM_DbgValueOp : LLVM_DbgIntrOp<"dbg.value", "value"> {
  let summary = "Describe the current value of a local debug info variable.";
  let arguments = (ins LLVM_Type:$value, LLVM_DILocalVariableAttr:$varInfo);
}

//
// Variadic function intrinsics.
//

def LLVM_VaStartOp : LLVM_ZeroResultIntrOp<"vastart">,
                     Arguments<(ins LLVM_i8Ptr:$arg_list)> {
  let assemblyFormat = "$arg_list attr-dict `:` qualified(type($arg_list))";
  let summary = "Initializes `arg_list` for subsequent variadic argument extractions.";
}

def LLVM_VaCopyOp : LLVM_ZeroResultIntrOp<"vacopy">,
                    Arguments<(ins LLVM_i8Ptr:$dest_list, LLVM_i8Ptr:$src_list)> {
  let assemblyFormat = "$src_list `to` $dest_list attr-dict `:` type(operands)";
  let summary = "Copies the current argument position from `src_list` to `dest_list`.";
}

def LLVM_VaEndOp : LLVM_ZeroResultIntrOp<"vaend">,
                   Arguments<(ins LLVM_i8Ptr:$arg_list)> {
  let assemblyFormat = "$arg_list attr-dict `:` qualified(type($arg_list))";
  let summary = "Destroys `arg_list`, which has been initialized by `intr.vastart` or `intr.vacopy`.";
}

//
// Exception handling intrinsics.
//

def LLVM_EhTypeidForOp : LLVM_OneResultIntrOp<"eh.typeid.for"> {
    let arguments = (ins LLVM_i8Ptr:$type_info);
    let assemblyFormat = "$type_info attr-dict `:` functional-type(operands, results)";
}

//
// Stack save/restore intrinsics.
//

def LLVM_StackSaveOp : LLVM_OneResultIntrOp<"stacksave"> {
  let assemblyFormat = "attr-dict `:` type($res)";
}

def LLVM_StackRestoreOp : LLVM_ZeroResultIntrOp<"stackrestore"> {
  let arguments = (ins LLVM_i8Ptr:$ptr);
  let assemblyFormat = "$ptr attr-dict `:` qualified(type($ptr))";
}

//
// Vector Reductions.
//

// LLVM vector reduction over a single vector.
class LLVM_VecReductionBase<string mnem, Type element>
    : LLVM_OneResultIntrOp<"vector.reduce." # mnem, [], [0],
                           [Pure, SameOperandsAndResultElementType]>,
      Arguments<(ins LLVM_VectorOf<element>)>;

class LLVM_VecReductionF<string mnem>
    : LLVM_VecReductionBase<mnem, AnyFloat>;

class LLVM_VecReductionI<string mnem>
    : LLVM_VecReductionBase<mnem, AnySignlessInteger>;

// LLVM vector reduction over a single vector, with an initial value,
// and with permission to reassociate the reduction operations.
class LLVM_VecReductionAccBase<string mnem, Type element>
    : LLVM_OneResultIntrOp<"vector.reduce." # mnem, [], [0],
                           [Pure, SameOperandsAndResultElementType]>,
      Arguments<(ins element:$start_value, LLVM_VectorOf<element>:$input,
                 DefaultValuedAttr<BoolAttr, "false">:$reassoc)> {
  let llvmBuilder = [{
    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_}] # mnem # [{,
        { }] # !interleave(ListIntSubst<LLVM_IntrPatterns.operand, [1]>.lst,
                           ", ") # [{
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    llvm::FastMathFlags origFM = builder.getFastMathFlags();
    llvm::FastMathFlags tempFM = origFM;
    tempFM.setAllowReassoc($reassoc);
    builder.setFastMathFlags(tempFM);  // set fastmath flag
    $res = builder.CreateCall(fn, operands);
    builder.setFastMathFlags(origFM);  // restore fastmath flag
  }];
  let mlirBuilder = [{
    bool allowReassoc = inst->getFastMathFlags().allowReassoc();
    $res = $_builder.create<$_qualCppClassName>($_location,
      $_resultType, $start_value, $input, allowReassoc);
  }];
}

class LLVM_VecReductionAccF<string mnem>
    : LLVM_VecReductionAccBase<mnem, AnyFloat>;

def LLVM_vector_reduce_add : LLVM_VecReductionI<"add">;
def LLVM_vector_reduce_and : LLVM_VecReductionI<"and">;
def LLVM_vector_reduce_mul : LLVM_VecReductionI<"mul">;
def LLVM_vector_reduce_or : LLVM_VecReductionI<"or">;
def LLVM_vector_reduce_smax : LLVM_VecReductionI<"smax">;
def LLVM_vector_reduce_smin : LLVM_VecReductionI<"smin">;
def LLVM_vector_reduce_umax : LLVM_VecReductionI<"umax">;
def LLVM_vector_reduce_umin : LLVM_VecReductionI<"umin">;
def LLVM_vector_reduce_xor : LLVM_VecReductionI<"xor">;

def LLVM_vector_reduce_fmax : LLVM_VecReductionF<"fmax">;
def LLVM_vector_reduce_fmin : LLVM_VecReductionF<"fmin">;

def LLVM_vector_reduce_fadd : LLVM_VecReductionAccF<"fadd">;
def LLVM_vector_reduce_fmul : LLVM_VecReductionAccF<"fmul">;

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
def LLVM_MatrixColumnMajorLoadOp : LLVM_OneResultIntrOp<"matrix.column.major.load"> {
  let arguments = (ins LLVM_AnyPointer:$data, AnySignlessInteger:$stride, I1Attr:$isVolatile,
                   I32Attr:$rows, I32Attr:$columns);
  let results = (outs LLVM_AnyVector:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "$data `,` `<` `stride` `=` $stride `>` attr-dict"
    "`:` type($res) `from` qualified(type($data)) `stride` type($stride)";

  string llvmBuilder = [{
    llvm::MatrixBuilder mb(builder);
    const llvm::DataLayout &dl =
      builder.GetInsertBlock()->getModule()->getDataLayout();
    llvm::Type *ElemTy = moduleTranslation.convertType(
        getVectorElementType(op.getType()));
    llvm::Align align = dl.getABITypeAlign(ElemTy);
    $res = mb.CreateColumnMajorLoad(
      ElemTy, $data, align, $stride, $isVolatile, $rows,
      $columns);
  }];
  string mlirBuilder = [{
    $res = $_builder.create<LLVM::MatrixColumnMajorLoadOp>(
      $_location, $_resultType, $data, $stride,
      $_int_attr($isVolatile), $_int_attr($rows), $_int_attr($columns));
  }];
}

/// Create a column major, strided 2-D matrix store, as specified in the LLVM
/// MatrixBuilder.
/// matrix     - Matrix to store
/// ptr        - Pointer to write back to
/// isVolatile - True if the load operation is marked as volatile.
/// rows       - Number of rows in matrix (must be a constant)
/// columns    - Number of columns in matrix (must be a constant)
/// stride     - Space between columns
def LLVM_MatrixColumnMajorStoreOp : LLVM_ZeroResultIntrOp<"matrix.column.major.store"> {
  let arguments = (ins LLVM_AnyVector:$matrix, LLVM_AnyPointer:$data,
                   AnySignlessInteger:$stride, I1Attr:$isVolatile, I32Attr:$rows,
                   I32Attr:$columns);
  let builders = [LLVM_VoidResultTypeOpBuilder, LLVM_ZeroResultOpBuilder];
  let assemblyFormat = "$matrix `,` $data `,` `<` `stride` `=` $stride `>` "
    "attr-dict`:` type($matrix) `to` qualified(type($data)) `stride` type($stride)";

  string llvmBuilder = [{
    llvm::MatrixBuilder mb(builder);
    const llvm::DataLayout &dl =
      builder.GetInsertBlock()->getModule()->getDataLayout();
    Type elementType = getVectorElementType(op.getMatrix().getType());
    llvm::Align align = dl.getABITypeAlign(
      moduleTranslation.convertType(elementType));
    mb.CreateColumnMajorStore(
      $matrix, $data, align, $stride, $isVolatile,
      $rows, $columns);
  }];
  string mlirBuilder = [{
    $_op = $_builder.create<LLVM::MatrixColumnMajorStoreOp>(
      $_location, $matrix, $data, $stride,
      $_int_attr($isVolatile), $_int_attr($rows), $_int_attr($columns));
  }];
}

/// Create a llvm.matrix.multiply call, multiplying 2-D matrices LHS and RHS, as
/// specified in the LLVM MatrixBuilder.
def LLVM_MatrixMultiplyOp : LLVM_OneResultIntrOp<"matrix.multiply"> {
  let arguments = (ins LLVM_AnyVector:$lhs, LLVM_AnyVector:$rhs, I32Attr:$lhs_rows,
                   I32Attr:$lhs_columns, I32Attr:$rhs_columns);
  let results = (outs LLVM_AnyVector:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "$lhs `,` $rhs attr-dict "
    "`:` `(` type($lhs) `,` type($rhs) `)` `->` type($res)";

  string llvmBuilder = [{
    llvm::MatrixBuilder mb(builder);
    $res = mb.CreateMatrixMultiply(
      $lhs, $rhs, $lhs_rows, $lhs_columns,
      $rhs_columns);
  }];
  string mlirBuilder = [{
    $res = $_builder.create<LLVM::MatrixMultiplyOp>(
      $_location, $_resultType, $lhs, $rhs,
      $_int_attr($lhs_rows), $_int_attr($lhs_columns), $_int_attr($rhs_columns));
  }];
}

/// Create a llvm.matrix.transpose call, transposing a `rows` x `columns` 2-D
/// `matrix`, as specified in the LLVM MatrixBuilder.
def LLVM_MatrixTransposeOp : LLVM_OneResultIntrOp<"matrix.transpose"> {
  let arguments = (ins LLVM_AnyVector:$matrix, I32Attr:$rows, I32Attr:$columns);
  let results = (outs LLVM_AnyVector:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat = "$matrix attr-dict `:` type($matrix) `into` type($res)";

  string llvmBuilder = [{
    llvm::MatrixBuilder mb(builder);
    $res = mb.CreateMatrixTranspose(
      $matrix, $rows, $columns);
  }];
  string mlirBuilder = [{
    $res = $_builder.create<LLVM::MatrixTransposeOp>(
      $_location, $_resultType, $matrix,
      $_int_attr($rows), $_int_attr($columns));
  }];
}

//
// LLVM masked operations.
//

/// Create a llvm.get.active.lane.mask to set a mask up to a given position.
def LLVM_GetActiveLaneMaskOp
    : LLVM_OneResultIntrOp<"get.active.lane.mask", [0], [0], [Pure]> {
  let arguments = (ins AnySignlessInteger:$base, AnySignlessInteger:$n);
  let assemblyFormat = "$base `,` $n attr-dict `:` "
    "type($base) `,` type($n) `to` type($res)";
}

/// Create a call to Masked Load intrinsic.
def LLVM_MaskedLoadOp : LLVM_OneResultIntrOp<"masked.load"> {
  let arguments = (ins LLVM_AnyPointer:$data, LLVM_VectorOf<I1>:$mask,
                   Variadic<LLVM_AnyVector>:$pass_thru, I32Attr:$alignment);
  let results = (outs LLVM_AnyVector:$res);
  let assemblyFormat =
    "operands attr-dict `:` functional-type(operands, results)";

  string llvmBuilder = [{
    $res = $pass_thru.empty() ? builder.CreateMaskedLoad(
        $_resultType, $data, llvm::Align($alignment), $mask) :
      builder.CreateMaskedLoad(
        $_resultType, $data, llvm::Align($alignment), $mask, $pass_thru[0]);
  }];
  string mlirBuilder = [{
    $res = $_builder.create<LLVM::MaskedLoadOp>($_location,
      $_resultType, $data, $mask, $pass_thru, $_int_attr($alignment));
  }];
  list<int> llvmArgIndices = [0, 2, 3, 1];
}

/// Create a call to Masked Store intrinsic.
def LLVM_MaskedStoreOp : LLVM_ZeroResultIntrOp<"masked.store"> {
  let arguments = (ins LLVM_AnyVector:$value, LLVM_AnyPointer:$data,
                   LLVM_VectorOf<I1>:$mask, I32Attr:$alignment);
  let builders = [LLVM_VoidResultTypeOpBuilder, LLVM_ZeroResultOpBuilder];
  let assemblyFormat = "$value `,` $data `,` $mask attr-dict `:` "
    "type($value) `,` type($mask) `into` qualified(type($data))";

  string llvmBuilder = [{
    builder.CreateMaskedStore(
      $value, $data, llvm::Align($alignment), $mask);
  }];
  string mlirBuilder = [{
    $_op = $_builder.create<LLVM::MaskedStoreOp>($_location,
      $value, $data, $mask, $_int_attr($alignment));
  }];
  list<int> llvmArgIndices = [0, 1, 3, 2];
}

/// Create a call to Masked Gather intrinsic.
def LLVM_masked_gather : LLVM_OneResultIntrOp<"masked.gather"> {
  let arguments = (ins LLVM_VectorOf<LLVM_AnyPointer>:$ptrs,
                   LLVM_VectorOf<I1>:$mask, Variadic<LLVM_AnyVector>:$pass_thru,
                   I32Attr:$alignment);
  let results = (outs LLVM_AnyVector:$res);
  let builders = [LLVM_OneResultOpBuilder];
  let assemblyFormat =
    "operands attr-dict `:` functional-type(operands, results)";

  string llvmBuilder = [{
    $res = $pass_thru.empty() ? builder.CreateMaskedGather(
        $_resultType, $ptrs, llvm::Align($alignment), $mask) :
      builder.CreateMaskedGather(
        $_resultType, $ptrs, llvm::Align($alignment), $mask, $pass_thru[0]);
  }];
  string mlirBuilder = [{
    $res = $_builder.create<LLVM::masked_gather>($_location,
      $_resultType, $ptrs, $mask, $pass_thru, $_int_attr($alignment));
  }];
  list<int> llvmArgIndices = [0, 2, 3, 1];
}

/// Create a call to Masked Scatter intrinsic.
def LLVM_masked_scatter : LLVM_ZeroResultIntrOp<"masked.scatter"> {
  let arguments = (ins LLVM_AnyVector:$value, LLVM_VectorOf<LLVM_AnyPointer>:$ptrs,
                   LLVM_VectorOf<I1>:$mask, I32Attr:$alignment);
  let builders = [LLVM_VoidResultTypeOpBuilder, LLVM_ZeroResultOpBuilder];
  let assemblyFormat = "$value `,` $ptrs `,` $mask attr-dict `:` "
    "type($value) `,` type($mask) `into` type($ptrs)";

  string llvmBuilder = [{
    builder.CreateMaskedScatter(
      $value, $ptrs, llvm::Align($alignment), $mask);
  }];
  string mlirBuilder = [{
    $_op = $_builder.create<LLVM::masked_scatter>($_location,
      $value, $ptrs, $mask, $_int_attr($alignment));
  }];
  list<int> llvmArgIndices = [0, 1, 3, 2];
}

/// Create a call to Masked Expand Load intrinsic.
def LLVM_masked_expandload : LLVM_IntrOp<"masked.expandload", [0], [], [], 1> {
  let arguments = (ins LLVM_AnyPointer, LLVM_VectorOf<I1>, LLVM_AnyVector);
}

/// Create a call to Masked Compress Store intrinsic.
def LLVM_masked_compressstore
    : LLVM_IntrOp<"masked.compressstore", [], [0], [], 0> {
  let arguments = (ins LLVM_AnyVector, LLVM_AnyPointer, LLVM_VectorOf<I1>);
}

/// Create a call to vscale intrinsic.
def LLVM_vscale : LLVM_IntrOp<"vscale", [0], [], [], 1>;

/// Create a call to stepvector intrinsic.
def LLVM_StepVectorOp
    : LLVM_IntrOp<"experimental.stepvector", [0], [], [Pure], 1> {
  let arguments = (ins);
  let results = (outs LLVM_VectorOf<AnySignlessInteger>:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

/// Create a call to vector.insert intrinsic
def LLVM_vector_insert
    : LLVM_Op<"intr.vector.insert",
                 [Pure, AllTypesMatch<["dstvec", "res"]>,
                  PredOpTrait<"vectors are not bigger than 2^17 bits.", And<[
                    CPred<"getSrcVectorBitWidth() <= 131072">,
                    CPred<"getDstVectorBitWidth() <= 131072">
                  ]>>,
                  PredOpTrait<"it is not inserting scalable into fixed-length vectors.",
                    CPred<"!isScalableVectorType($srcvec.getType()) || "
                          "isScalableVectorType($dstvec.getType())">>]> {
  let arguments = (ins LLVM_AnyVector:$srcvec, LLVM_AnyVector:$dstvec,
                       I64Attr:$pos);
  let results = (outs LLVM_AnyVector:$res);
  let builders = [LLVM_OneResultOpBuilder];
  string llvmBuilder = [{
    $res = builder.CreateInsertVector(
        $_resultType, $dstvec, $srcvec, builder.getInt64($pos));
  }];
  let assemblyFormat = "$srcvec `,` $dstvec `[` $pos `]` attr-dict `:` "
    "type($srcvec) `into` type($res)";
  let extraClassDeclaration = [{
    uint64_t getVectorBitWidth(Type vector) {
      return getVectorNumElements(vector).getKnownMinValue() *
             getVectorElementType(vector).getIntOrFloatBitWidth();
    }
    uint64_t getSrcVectorBitWidth() {
      return getVectorBitWidth(getSrcvec().getType());
    }
    uint64_t getDstVectorBitWidth() {
      return getVectorBitWidth(getDstvec().getType());
    }
  }];
}

/// Create a call to vector.extract intrinsic
def LLVM_vector_extract
    : LLVM_Op<"intr.vector.extract",
                 [Pure,
                  PredOpTrait<"vectors are not bigger than 2^17 bits.", And<[
                    CPred<"getSrcVectorBitWidth() <= 131072">,
                    CPred<"getResVectorBitWidth() <= 131072">
                  ]>>,
                  PredOpTrait<"it is not extracting scalable from fixed-length vectors.",
                    CPred<"!isScalableVectorType($res.getType()) || "
                          "isScalableVectorType($srcvec.getType())">>]> {
  let arguments = (ins LLVM_AnyVector:$srcvec, I64Attr:$pos);
  let results = (outs LLVM_AnyVector:$res);
  let builders = [LLVM_OneResultOpBuilder];
  string llvmBuilder = [{
    $res = builder.CreateExtractVector(
        $_resultType, $srcvec, builder.getInt64($pos));
  }];
  let assemblyFormat = "$srcvec `[` $pos `]` attr-dict `:` "
    "type($res) `from` type($srcvec)";
  let extraClassDeclaration = [{
    uint64_t getVectorBitWidth(Type vector) {
      return getVectorNumElements(vector).getKnownMinValue() *
             getVectorElementType(vector).getIntOrFloatBitWidth();
    }
    uint64_t getSrcVectorBitWidth() {
      return getVectorBitWidth(getSrcvec().getType());
    }
    uint64_t getResVectorBitWidth() {
      return getVectorBitWidth(getRes().getType());
    }
  }];
}

//===--------------------------------------------------------------------===//
// CallIntrinsicOp
//===--------------------------------------------------------------------===//

def LLVM_CallIntrinsicOp : LLVM_Op<"call_intrinsic"> {
  let summary = "Call to an LLVM intrinsic function.";
  let description = [{
    Call the specified llvm intrinsic. If the intrinsic is overloaded, use
    the MLIR function type of this op to determine which intrinsic to call.
    }];
  let arguments = (ins StrAttr:$intrin, Variadic<LLVM_Type>:$args);
  let results = (outs Variadic<LLVM_Type>:$results);
  let llvmBuilder = [{
    return convertCallLLVMIntrinsicOp(op, builder, moduleTranslation);
  }];
  let assemblyFormat = [{
    $intrin `(` $args `)` `:` functional-type($args, $results) attr-dict
  }];
}

//
// LLVM Vector Predication operations.
//

class LLVM_VPBinaryBase<string mnem, Type element>
    : LLVM_OneResultIntrOp<"vp." # mnem, [0], [], [Pure]>,
      Arguments<(ins LLVM_VectorOf<element>:$lhs, LLVM_VectorOf<element>:$rhs,
                     LLVM_VectorOf<I1>:$mask, I32:$evl)>;

class LLVM_VPBinaryI<string mnem> : LLVM_VPBinaryBase<mnem, AnySignlessInteger>;

class LLVM_VPBinaryF<string mnem> : LLVM_VPBinaryBase<mnem, AnyFloat>;

class LLVM_VPUnaryBase<string mnem, Type element>
    : LLVM_OneResultIntrOp<"vp." # mnem, [0], [], [Pure]>,
      Arguments<(ins LLVM_VectorOf<element>:$op,
                     LLVM_VectorOf<I1>:$mask, I32:$evl)>;

class LLVM_VPUnaryF<string mnem> : LLVM_VPUnaryBase<mnem, AnyFloat>;

class LLVM_VPTernaryBase<string mnem, Type element>
    : LLVM_OneResultIntrOp<"vp." # mnem, [0], [], [Pure]>,
      Arguments<(ins LLVM_VectorOf<element>:$op1, LLVM_VectorOf<element>:$op2,
                     LLVM_VectorOf<element>:$op3, LLVM_VectorOf<I1>:$mask,
                     I32:$evl)>;

class LLVM_VPTernaryF<string mnem> : LLVM_VPTernaryBase<mnem, AnyFloat>;

class LLVM_VPReductionBase<string mnem, Type element>
    : LLVM_OneResultIntrOp<"vp.reduce." # mnem, [], [1], [Pure]>,
      Arguments<(ins element:$satrt_value, LLVM_VectorOf<element>:$val,
                     LLVM_VectorOf<I1>:$mask, I32:$evl)>;

class LLVM_VPReductionI<string mnem> : LLVM_VPReductionBase<mnem, AnySignlessInteger>;

class LLVM_VPReductionF<string mnem> : LLVM_VPReductionBase<mnem, AnyFloat>;

class LLVM_VPSelectBase<string mnem>
    : LLVM_OneResultIntrOp<"vp." # mnem, [], [1], [Pure]>,
      Arguments<(ins LLVM_VectorOf<I1>:$cond, LLVM_AnyVector:$true_val,
                     LLVM_AnyVector:$false_val, I32:$evl)>;

class LLVM_VPCastBase<string mnem, Type element>
    : LLVM_OneResultIntrOp<"vp." # mnem, [0], [0], [Pure]>,
      Arguments<(ins LLVM_VectorOf<element>:$src,
                     LLVM_VectorOf<I1>:$mask, I32:$evl)>;

class LLVM_VPCastI<string mnem>   : LLVM_VPCastBase<mnem, AnySignlessInteger>;

class LLVM_VPCastF<string mnem>   : LLVM_VPCastBase<mnem, AnyFloat>;

class LLVM_VPCastPtr<string mnem> : LLVM_VPCastBase<mnem, LLVM_AnyPointer>;

// Integer Binary
def LLVM_VPAddOp  : LLVM_VPBinaryI<"add">;
def LLVM_VPSubOp  : LLVM_VPBinaryI<"sub">;
def LLVM_VPMulOp  : LLVM_VPBinaryI<"mul">;
def LLVM_VPSDivOp : LLVM_VPBinaryI<"sdiv">;
def LLVM_VPUDivOp : LLVM_VPBinaryI<"udiv">;
def LLVM_VPSRemOp : LLVM_VPBinaryI<"srem">;
def LLVM_VPURemOp : LLVM_VPBinaryI<"urem">;
def LLVM_VPAShrOp : LLVM_VPBinaryI<"ashr">;
def LLVM_VPLShrOp : LLVM_VPBinaryI<"lshr">;
def LLVM_VPShlOp  : LLVM_VPBinaryI<"shl">;
def LLVM_VPOrOp   : LLVM_VPBinaryI<"or">;
def LLVM_VPAndOp  : LLVM_VPBinaryI<"and">;
def LLVM_VPXorOp  : LLVM_VPBinaryI<"xor">;

// Float Binary
def LLVM_VPFAddOp : LLVM_VPBinaryF<"fadd">;
def LLVM_VPFSubOp : LLVM_VPBinaryF<"fsub">;
def LLVM_VPFMulOp : LLVM_VPBinaryF<"fmul">;
def LLVM_VPFDivOp : LLVM_VPBinaryF<"fdiv">;
def LLVM_VPFRemOp : LLVM_VPBinaryF<"frem">;

// Float Unary
def LLVM_VPFNegOp : LLVM_VPUnaryF<"fneg">;

// Float Ternary
def LLVM_VPFMulAddOp  : LLVM_VPTernaryF<"fmuladd">;
def LLVM_VPFmaOp      : LLVM_VPTernaryF<"fma">;

// Integer Reduction
def LLVM_VPReduceAddOp  : LLVM_VPReductionI<"add">;
def LLVM_VPReduceMulOp  : LLVM_VPReductionI<"mul">;
def LLVM_VPReduceAndOp  : LLVM_VPReductionI<"and">;
def LLVM_VPReduceOrOp   : LLVM_VPReductionI<"or">;
def LLVM_VPReduceXorOp  : LLVM_VPReductionI<"xor">;
def LLVM_VPReduceSMaxOp : LLVM_VPReductionI<"smax">;
def LLVM_VPReduceSMinOp : LLVM_VPReductionI<"smin">;
def LLVM_VPReduceUMaxOp : LLVM_VPReductionI<"umax">;
def LLVM_VPReduceUMinOp : LLVM_VPReductionI<"umin">;

// Float Reduction
def LLVM_VPReduceFAddOp : LLVM_VPReductionF<"fadd">;
def LLVM_VPReduceFMulOp : LLVM_VPReductionF<"fmul">;
def LLVM_VPReduceFMaxOp : LLVM_VPReductionF<"fmax">;
def LLVM_VPReduceFMinOp : LLVM_VPReductionF<"fmin">;

def LLVM_VPSelectMinOp : LLVM_VPSelectBase<"select">;
def LLVM_VPMergeMinOp  : LLVM_VPSelectBase<"merge">;

// Load/store
def LLVM_VPLoadOp
    : LLVM_OneResultIntrOp<"vp.load", [0], [0], []>,
      Arguments<(ins LLVM_AnyPointer:$ptr,
                     LLVM_VectorOf<I1>:$mask, I32:$evl)>;

def LLVM_VPStoreOp
    : LLVM_ZeroResultIntrOp<"vp.store", [0, 1], []>,
      Arguments<(ins LLVM_AnyVector:$val,
                     LLVM_AnyPointer:$ptr,
                     LLVM_VectorOf<I1>:$mask, I32:$evl)>;

// Strided load/store
def LLVM_VPStridedLoadOp
    : LLVM_OneResultIntrOp<"experimental.vp.strided.load", [0], [0, 1], []>,
      Arguments<(ins LLVM_AnyPointer:$ptr, AnySignlessInteger:$stride,
                     LLVM_VectorOf<I1>:$mask, I32:$evl)>;

def LLVM_VPStridedStoreOp
    : LLVM_ZeroResultIntrOp<"experimental.vp.strided.store",[0, 1, 2], []>,
      Arguments<(ins LLVM_AnyVector:$val, LLVM_AnyPointer:$ptr,
                     AnySignlessInteger:$stride, LLVM_VectorOf<I1>:$mask, I32:$evl)>;

def LLVM_VPTruncOp : LLVM_VPCastI<"trunc">;
def LLVM_VPZExtOp  : LLVM_VPCastI<"zext">;
def LLVM_VPSExtOp  : LLVM_VPCastI<"sext">;

def LLVM_VPFPTruncOp : LLVM_VPCastF<"fptrunc">;
def LLVM_VPFPExtOp   : LLVM_VPCastF<"fpext">;

def LLVM_VPFPToUIOp : LLVM_VPCastF<"fptoui">;
def LLVM_VPFPToSIOp : LLVM_VPCastF<"fptosi">;

def LLVM_VPUIToFPOp : LLVM_VPCastI<"uitofp">;
def LLVM_VPSIToFPOp : LLVM_VPCastI<"sitofp">;

def LLVM_VPPtrToIntOp : LLVM_VPCastPtr<"ptrtoint">;
def LLVM_VPIntToPtrOp : LLVM_VPCastI<"inttoptr">;

#endif // LLVM_INTRINSIC_OP
