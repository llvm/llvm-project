// WebAssemblyInstrInfo.td-Describe the WebAssembly Instructions-*- tablegen -*-
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// WebAssembly Instruction definitions.
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// WebAssembly Instruction Predicate Definitions.
//===----------------------------------------------------------------------===//

def IsPIC     : Predicate<"TM.isPositionIndependent()">;
def IsNotPIC  : Predicate<"!TM.isPositionIndependent()">;

def HasAddr32 : Predicate<"!Subtarget->hasAddr64()">;

def HasAddr64 : Predicate<"Subtarget->hasAddr64()">;

def HasSIMD128 :
    Predicate<"Subtarget->hasSIMD128()">,
    AssemblerPredicate<(all_of FeatureSIMD128), "simd128">;

def HasUnimplementedSIMD128 :
    Predicate<"Subtarget->hasUnimplementedSIMD128()">,
    AssemblerPredicate<(all_of FeatureUnimplementedSIMD128), "unimplemented-simd128">;

def HasAtomics :
    Predicate<"Subtarget->hasAtomics()">,
    AssemblerPredicate<(all_of FeatureAtomics), "atomics">;

def HasMultivalue :
    Predicate<"Subtarget->hasMultivalue()">,
    AssemblerPredicate<(all_of FeatureMultivalue), "multivalue">;

def HasNontrappingFPToInt :
    Predicate<"Subtarget->hasNontrappingFPToInt()">,
    AssemblerPredicate<(all_of FeatureNontrappingFPToInt), "nontrapping-fptoint">;

def NotHasNontrappingFPToInt :
    Predicate<"!Subtarget->hasNontrappingFPToInt()">,
    AssemblerPredicate<(all_of (not FeatureNontrappingFPToInt)), "nontrapping-fptoint">;

def HasSignExt :
    Predicate<"Subtarget->hasSignExt()">,
    AssemblerPredicate<(all_of FeatureSignExt), "sign-ext">;

def HasTailCall :
    Predicate<"Subtarget->hasTailCall()">,
    AssemblerPredicate<(all_of FeatureTailCall), "tail-call">;

def HasExceptionHandling :
    Predicate<"Subtarget->hasExceptionHandling()">,
    AssemblerPredicate<(all_of FeatureExceptionHandling), "exception-handling">;

def HasBulkMemory :
    Predicate<"Subtarget->hasBulkMemory()">,
    AssemblerPredicate<(all_of FeatureBulkMemory), "bulk-memory">;

def HasReferenceTypes :
    Predicate<"Subtarget->hasReferenceTypes()">,
    AssemblerPredicate<(all_of FeatureReferenceTypes), "reference-types">;

//===----------------------------------------------------------------------===//
// WebAssembly-specific DAG Node Types.
//===----------------------------------------------------------------------===//

def SDT_WebAssemblyCallSeqStart : SDCallSeqStart<[SDTCisVT<0, iPTR>,
                                                  SDTCisVT<1, iPTR>]>;
def SDT_WebAssemblyCallSeqEnd :
    SDCallSeqEnd<[SDTCisVT<0, iPTR>, SDTCisVT<1, iPTR>]>;
def SDT_WebAssemblyCall0      : SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>;
def SDT_WebAssemblyCall1      : SDTypeProfile<1, -1, [SDTCisPtrTy<1>]>;
def SDT_WebAssemblyBrTable    : SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>;
def SDT_WebAssemblyArgument   : SDTypeProfile<1, 1, [SDTCisVT<1, i32>]>;
def SDT_WebAssemblyReturn     : SDTypeProfile<0, -1, []>;
def SDT_WebAssemblyWrapper    : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>,
                                                     SDTCisPtrTy<0>]>;
def SDT_WebAssemblyWrapperPIC : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>,
                                                     SDTCisPtrTy<0>]>;
def SDT_WebAssemblyThrow      : SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>;

//===----------------------------------------------------------------------===//
// WebAssembly-specific DAG Nodes.
//===----------------------------------------------------------------------===//

def WebAssemblycallseq_start :
    SDNode<"ISD::CALLSEQ_START", SDT_WebAssemblyCallSeqStart,
           [SDNPHasChain, SDNPOutGlue]>;
def WebAssemblycallseq_end :
    SDNode<"ISD::CALLSEQ_END", SDT_WebAssemblyCallSeqEnd,
           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;
def WebAssemblybr_table : SDNode<"WebAssemblyISD::BR_TABLE",
                                 SDT_WebAssemblyBrTable,
                                 [SDNPHasChain, SDNPVariadic]>;
def WebAssemblyargument : SDNode<"WebAssemblyISD::ARGUMENT",
                                 SDT_WebAssemblyArgument>;
def WebAssemblyreturn   : SDNode<"WebAssemblyISD::RETURN",
                                 SDT_WebAssemblyReturn,
                                 [SDNPHasChain, SDNPVariadic]>;
def WebAssemblywrapper  : SDNode<"WebAssemblyISD::Wrapper",
                                 SDT_WebAssemblyWrapper>;
def WebAssemblywrapperPIC  : SDNode<"WebAssemblyISD::WrapperPIC",
                                     SDT_WebAssemblyWrapperPIC>;
def WebAssemblythrow : SDNode<"WebAssemblyISD::THROW", SDT_WebAssemblyThrow,
                              [SDNPHasChain, SDNPVariadic]>;

//===----------------------------------------------------------------------===//
// WebAssembly-specific Operands.
//===----------------------------------------------------------------------===//

// Default Operand has AsmOperandClass "Imm" which is for integers (and
// symbols), so specialize one for floats:
def FPImmAsmOperand : AsmOperandClass {
  let Name = "FPImm";
  let PredicateMethod = "isFPImm";
}

class FPOperand<ValueType ty> : Operand<ty> {
  AsmOperandClass ParserMatchClass = FPImmAsmOperand;
}

let OperandNamespace = "WebAssembly" in {

let OperandType = "OPERAND_BASIC_BLOCK" in
def bb_op : Operand<OtherVT>;

let OperandType = "OPERAND_LOCAL" in
def local_op : Operand<i32>;

let OperandType = "OPERAND_GLOBAL" in
def global_op : Operand<i32>;

let OperandType = "OPERAND_I32IMM" in
def i32imm_op : Operand<i32>;

let OperandType = "OPERAND_I64IMM" in
def i64imm_op : Operand<i64>;

let OperandType = "OPERAND_F32IMM" in
def f32imm_op : FPOperand<f32>;

let OperandType = "OPERAND_F64IMM" in
def f64imm_op : FPOperand<f64>;

let OperandType = "OPERAND_VEC_I8IMM" in
def vec_i8imm_op : Operand<i32>;

let OperandType = "OPERAND_VEC_I16IMM" in
def vec_i16imm_op : Operand<i32>;

let OperandType = "OPERAND_VEC_I32IMM" in
def vec_i32imm_op : Operand<i32>;

let OperandType = "OPERAND_VEC_I64IMM" in
def vec_i64imm_op : Operand<i64>;

let OperandType = "OPERAND_FUNCTION32" in
def function32_op : Operand<i32>;

let OperandType = "OPERAND_TABLE" in
def table32_op : Operand<i32>;

let OperandType = "OPERAND_OFFSET32" in
def offset32_op : Operand<i32>;

let OperandType = "OPERAND_OFFSET64" in
def offset64_op : Operand<i64>;

let OperandType = "OPERAND_P2ALIGN" in {
def P2Align : Operand<i32> {
  let PrintMethod = "printWebAssemblyP2AlignOperand";
}

let OperandType = "OPERAND_EVENT" in
def event_op : Operand<i32>;

} // OperandType = "OPERAND_P2ALIGN"

let OperandType = "OPERAND_SIGNATURE" in
def Signature : Operand<i32> {
  let PrintMethod = "printWebAssemblySignatureOperand";
}

let OperandType = "OPERAND_HEAPTYPE" in
def HeapType : Operand<i32> {
  let PrintMethod = "printWebAssemblyHeapTypeOperand";
}

let OperandType = "OPERAND_TYPEINDEX" in
def TypeIndex : Operand<i32>;

} // OperandNamespace = "WebAssembly"

//===----------------------------------------------------------------------===//
// WebAssembly Register to Stack instruction mapping
//===----------------------------------------------------------------------===//

class StackRel;
def getStackOpcode : InstrMapping {
  let FilterClass = "StackRel";
  let RowFields = ["BaseName"];
  let ColFields = ["StackBased"];
  let KeyCol = ["false"];
  let ValueCols = [["true"]];
}

//===----------------------------------------------------------------------===//
// WebAssembly 32 to 64-bit instruction mapping
//===----------------------------------------------------------------------===//

class Wasm64Rel;
def getWasm64Opcode : InstrMapping {
  let FilterClass = "Wasm64Rel";
  let RowFields = ["Wasm32Name"];
  let ColFields = ["IsWasm64"];
  let KeyCol = ["false"];
  let ValueCols = [["true"]];
}

//===----------------------------------------------------------------------===//
// WebAssembly Instruction Format Definitions.
//===----------------------------------------------------------------------===//

include "WebAssemblyInstrFormats.td"

//===----------------------------------------------------------------------===//
// Additional instructions.
//===----------------------------------------------------------------------===//

multiclass ARGUMENT<WebAssemblyRegClass reg, ValueType vt> {
  let hasSideEffects = 1, isCodeGenOnly = 1, Defs = []<Register>,
      Uses = [ARGUMENTS] in
  defm ARGUMENT_#vt :
    I<(outs reg:$res), (ins i32imm:$argno), (outs), (ins i32imm:$argno),
      [(set (vt reg:$res), (WebAssemblyargument timm:$argno))]>;
}
defm "": ARGUMENT<I32, i32>;
defm "": ARGUMENT<I64, i64>;
defm "": ARGUMENT<F32, f32>;
defm "": ARGUMENT<F64, f64>;
defm "": ARGUMENT<EXNREF, exnref>;
defm "": ARGUMENT<FUNCREF, funcref>;
defm "": ARGUMENT<EXTERNREF, externref>;

// local.get and local.set are not generated by instruction selection; they
// are implied by virtual register uses and defs.
multiclass LOCAL<WebAssemblyRegClass vt> {
  let hasSideEffects = 0 in {
  // COPY is not an actual instruction in wasm, but since we allow local.get and
  // local.set to be implicit during most of codegen, we can have a COPY which
  // is actually a no-op because all the work is done in the implied local.get
  // and local.set. COPYs are eliminated (and replaced with
  // local.get/local.set) in the ExplicitLocals pass.
  let isAsCheapAsAMove = 1, isCodeGenOnly = 1 in
  defm COPY_#vt : I<(outs vt:$res), (ins vt:$src), (outs), (ins), [],
                    "local.copy\t$res, $src", "local.copy">;

  // TEE is similar to COPY, but writes two copies of its result. Typically
  // this would be used to stackify one result and write the other result to a
  // local.
  let isAsCheapAsAMove = 1, isCodeGenOnly = 1 in
  defm TEE_#vt : I<(outs vt:$res, vt:$also), (ins vt:$src), (outs), (ins), [],
                   "local.tee\t$res, $also, $src", "local.tee">;

  // This is the actual local.get instruction in wasm. These are made explicit
  // by the ExplicitLocals pass. It has mayLoad because it reads from a wasm
  // local, which is a side effect not otherwise modeled in LLVM.
  let mayLoad = 1, isAsCheapAsAMove = 1 in
  defm LOCAL_GET_#vt : I<(outs vt:$res), (ins local_op:$local),
                         (outs), (ins local_op:$local), [],
                         "local.get\t$res, $local", "local.get\t$local", 0x20>;

  // This is the actual local.set instruction in wasm. These are made explicit
  // by the ExplicitLocals pass. It has mayStore because it writes to a wasm
  // local, which is a side effect not otherwise modeled in LLVM.
  let mayStore = 1, isAsCheapAsAMove = 1 in
  defm LOCAL_SET_#vt : I<(outs), (ins local_op:$local, vt:$src),
                         (outs), (ins local_op:$local), [],
                         "local.set\t$local, $src", "local.set\t$local", 0x21>;

  // This is the actual local.tee instruction in wasm. TEEs are turned into
  // LOCAL_TEEs by the ExplicitLocals pass. It has mayStore for the same reason
  // as LOCAL_SET.
  let mayStore = 1, isAsCheapAsAMove = 1 in
  defm LOCAL_TEE_#vt : I<(outs vt:$res), (ins local_op:$local, vt:$src),
                         (outs), (ins local_op:$local), [],
                         "local.tee\t$res, $local, $src", "local.tee\t$local",
                         0x22>;

  // Unused values must be dropped in some contexts.
  defm DROP_#vt : I<(outs), (ins vt:$src), (outs), (ins), [],
                    "drop\t$src", "drop", 0x1a>;

  let mayLoad = 1 in
  defm GLOBAL_GET_#vt : I<(outs vt:$res), (ins global_op:$local),
                          (outs), (ins global_op:$local), [],
                          "global.get\t$res, $local", "global.get\t$local",
                          0x23>;

  let mayStore = 1 in
  defm GLOBAL_SET_#vt : I<(outs), (ins global_op:$local, vt:$src),
                          (outs), (ins global_op:$local), [],
                          "global.set\t$local, $src", "global.set\t$local",
                          0x24>;

} // hasSideEffects = 0
}
defm "" : LOCAL<I32>;
defm "" : LOCAL<I64>;
defm "" : LOCAL<F32>;
defm "" : LOCAL<F64>;
defm "" : LOCAL<V128>, Requires<[HasSIMD128]>;
defm "" : LOCAL<EXNREF>, Requires<[HasExceptionHandling]>;
defm "" : LOCAL<FUNCREF>, Requires<[HasReferenceTypes]>;
defm "" : LOCAL<EXTERNREF>, Requires<[HasReferenceTypes]>;

let isMoveImm = 1, isAsCheapAsAMove = 1, isReMaterializable = 1 in {
defm CONST_I32 : I<(outs I32:$res), (ins i32imm_op:$imm),
                   (outs), (ins i32imm_op:$imm),
                   [(set I32:$res, imm:$imm)],
                   "i32.const\t$res, $imm", "i32.const\t$imm", 0x41>;
defm CONST_I64 : I<(outs I64:$res), (ins i64imm_op:$imm),
                   (outs), (ins i64imm_op:$imm),
                   [(set I64:$res, imm:$imm)],
                   "i64.const\t$res, $imm", "i64.const\t$imm", 0x42>;
defm CONST_F32 : I<(outs F32:$res), (ins f32imm_op:$imm),
                   (outs), (ins f32imm_op:$imm),
                   [(set F32:$res, fpimm:$imm)],
                   "f32.const\t$res, $imm", "f32.const\t$imm", 0x43>;
defm CONST_F64 : I<(outs F64:$res), (ins f64imm_op:$imm),
                   (outs), (ins f64imm_op:$imm),
                   [(set F64:$res, fpimm:$imm)],
                   "f64.const\t$res, $imm", "f64.const\t$imm", 0x44>;
} // isMoveImm = 1, isAsCheapAsAMove = 1, isReMaterializable = 1

def : Pat<(i32 (WebAssemblywrapper tglobaladdr:$addr)),
          (CONST_I32 tglobaladdr:$addr)>, Requires<[IsNotPIC, HasAddr32]>;
def : Pat<(i64 (WebAssemblywrapper tglobaladdr:$addr)),
          (CONST_I64 tglobaladdr:$addr)>, Requires<[IsNotPIC, HasAddr64]>;

def : Pat<(i32 (WebAssemblywrapper tglobaladdr:$addr)),
          (GLOBAL_GET_I32 tglobaladdr:$addr)>, Requires<[IsPIC, HasAddr32]>;

def : Pat<(i32 (WebAssemblywrapperPIC tglobaladdr:$addr)),
          (CONST_I32 tglobaladdr:$addr)>, Requires<[IsPIC, HasAddr32]>;
def : Pat<(i64 (WebAssemblywrapperPIC tglobaladdr:$addr)),
          (CONST_I64 tglobaladdr:$addr)>, Requires<[IsPIC, HasAddr64]>;

def : Pat<(i32 (WebAssemblywrapper tglobaltlsaddr:$addr)),
          (CONST_I32 tglobaltlsaddr:$addr)>, Requires<[HasAddr32]>;
def : Pat<(i64 (WebAssemblywrapper tglobaltlsaddr:$addr)),
          (CONST_I64 tglobaltlsaddr:$addr)>, Requires<[HasAddr64]>;

def : Pat<(i32 (WebAssemblywrapper texternalsym:$addr)),
          (GLOBAL_GET_I32 texternalsym:$addr)>, Requires<[IsPIC, HasAddr32]>;

def : Pat<(i32 (WebAssemblywrapper texternalsym:$addr)),
          (CONST_I32 texternalsym:$addr)>, Requires<[IsNotPIC, HasAddr32]>;
def : Pat<(i64 (WebAssemblywrapper texternalsym:$addr)),
          (CONST_I64 texternalsym:$addr)>, Requires<[IsNotPIC, HasAddr64]>;

def : Pat<(i32 (WebAssemblywrapper mcsym:$sym)), (CONST_I32 mcsym:$sym)>;
def : Pat<(i64 (WebAssemblywrapper mcsym:$sym)), (CONST_I64 mcsym:$sym)>;

//===----------------------------------------------------------------------===//
// Additional sets of instructions.
//===----------------------------------------------------------------------===//

include "WebAssemblyInstrMemory.td"
include "WebAssemblyInstrCall.td"
include "WebAssemblyInstrControl.td"
include "WebAssemblyInstrInteger.td"
include "WebAssemblyInstrConv.td"
include "WebAssemblyInstrFloat.td"
include "WebAssemblyInstrAtomics.td"
include "WebAssemblyInstrSIMD.td"
include "WebAssemblyInstrRef.td"
include "WebAssemblyInstrBulkMemory.td"
include "WebAssemblyInstrTable.td"
