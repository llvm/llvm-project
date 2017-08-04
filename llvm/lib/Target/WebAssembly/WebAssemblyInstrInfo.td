// WebAssemblyInstrInfo.td-Describe the WebAssembly Instructions-*- tablegen -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief WebAssembly Instruction definitions.
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// WebAssembly Instruction Predicate Definitions.
//===----------------------------------------------------------------------===//

def HasAddr32 : Predicate<"!Subtarget->hasAddr64()">;
def HasAddr64 : Predicate<"Subtarget->hasAddr64()">;
def HasSIMD128 : Predicate<"Subtarget->hasSIMD128()">,
                           AssemblerPredicate<"FeatureSIMD128", "simd128">;

//===----------------------------------------------------------------------===//
// WebAssembly-specific DAG Node Types.
//===----------------------------------------------------------------------===//

def SDT_WebAssemblyCallSeqStart : SDCallSeqStart<[SDTCisVT<0, iPTR>,
                                                  SDTCisVT<1, iPTR>]>;
def SDT_WebAssemblyCallSeqEnd :
    SDCallSeqEnd<[SDTCisVT<0, iPTR>, SDTCisVT<1, iPTR>]>;
def SDT_WebAssemblyCall0    : SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>;
def SDT_WebAssemblyCall1    : SDTypeProfile<1, -1, [SDTCisPtrTy<1>]>;
def SDT_WebAssemblyBrTable  : SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>;
def SDT_WebAssemblyArgument : SDTypeProfile<1, 1, [SDTCisVT<1, i32>]>;
def SDT_WebAssemblyReturn   : SDTypeProfile<0, -1, []>;
def SDT_WebAssemblyWrapper  : SDTypeProfile<1, 1, [SDTCisSameAs<0, 1>,
                                                   SDTCisPtrTy<0>]>;

//===----------------------------------------------------------------------===//
// WebAssembly-specific DAG Nodes.
//===----------------------------------------------------------------------===//

def WebAssemblycallseq_start :
    SDNode<"ISD::CALLSEQ_START", SDT_WebAssemblyCallSeqStart,
           [SDNPHasChain, SDNPOutGlue]>;
def WebAssemblycallseq_end :
    SDNode<"ISD::CALLSEQ_END", SDT_WebAssemblyCallSeqEnd,
           [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]>;
def WebAssemblycall0 : SDNode<"WebAssemblyISD::CALL0",
                              SDT_WebAssemblyCall0,
                              [SDNPHasChain, SDNPVariadic]>;
def WebAssemblycall1 : SDNode<"WebAssemblyISD::CALL1",
                              SDT_WebAssemblyCall1,
                              [SDNPHasChain, SDNPVariadic]>;
def WebAssemblybr_table : SDNode<"WebAssemblyISD::BR_TABLE",
                                 SDT_WebAssemblyBrTable,
                                 [SDNPHasChain, SDNPVariadic]>;
def WebAssemblyargument : SDNode<"WebAssemblyISD::ARGUMENT",
                                 SDT_WebAssemblyArgument>;
def WebAssemblyreturn   : SDNode<"WebAssemblyISD::RETURN",
                                 SDT_WebAssemblyReturn, [SDNPHasChain]>;
def WebAssemblywrapper  : SDNode<"WebAssemblyISD::Wrapper",
                                 SDT_WebAssemblyWrapper>;

//===----------------------------------------------------------------------===//
// WebAssembly-specific Operands.
//===----------------------------------------------------------------------===//

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
def f32imm_op : Operand<f32>;

let OperandType = "OPERAND_F64IMM" in
def f64imm_op : Operand<f64>;

let OperandType = "OPERAND_FUNCTION32" in
def function32_op : Operand<i32>;

let OperandType = "OPERAND_OFFSET32" in
def offset32_op : Operand<i32>;

let OperandType = "OPERAND_P2ALIGN" in {
def P2Align : Operand<i32> {
  let PrintMethod = "printWebAssemblyP2AlignOperand";
}
} // OperandType = "OPERAND_P2ALIGN"

let OperandType = "OPERAND_SIGNATURE" in {
def Signature : Operand<i32> {
  let PrintMethod = "printWebAssemblySignatureOperand";
}
} // OperandType = "OPERAND_SIGNATURE"

let OperandType = "OPERAND_TYPEINDEX" in
def TypeIndex : Operand<i32>;

} // OperandNamespace = "WebAssembly"

//===----------------------------------------------------------------------===//
// WebAssembly Instruction Format Definitions.
//===----------------------------------------------------------------------===//

include "WebAssemblyInstrFormats.td"

//===----------------------------------------------------------------------===//
// Additional instructions.
//===----------------------------------------------------------------------===//

multiclass ARGUMENT<WebAssemblyRegClass vt> {
  let hasSideEffects = 1, Uses = [ARGUMENTS], isCodeGenOnly = 1 in
  def ARGUMENT_#vt : I<(outs vt:$res), (ins i32imm:$argno),
                       [(set vt:$res, (WebAssemblyargument timm:$argno))]>;
}
multiclass SIMD_ARGUMENT<ValueType vt> {
  let hasSideEffects = 1, Uses = [ARGUMENTS], isCodeGenOnly = 1 in
  def ARGUMENT_#vt : SIMD_I<(outs V128:$res), (ins i32imm:$argno),
                            [(set (vt V128:$res),
                                  (WebAssemblyargument timm:$argno))]>;
}
defm : ARGUMENT<I32>;
defm : ARGUMENT<I64>;
defm : ARGUMENT<F32>;
defm : ARGUMENT<F64>;
defm : SIMD_ARGUMENT<v16i8>;
defm : SIMD_ARGUMENT<v8i16>;
defm : SIMD_ARGUMENT<v4i32>;
defm : SIMD_ARGUMENT<v4f32>;

let Defs = [ARGUMENTS] in {

// get_local and set_local are not generated by instruction selection; they
// are implied by virtual register uses and defs.
multiclass LOCAL<WebAssemblyRegClass vt> {
let hasSideEffects = 0 in {
  // COPY is not an actual instruction in wasm, but since we allow get_local and
  // set_local to be implicit during most of codegen, we can have a COPY which
  // is actually a no-op because all the work is done in the implied get_local
  // and set_local. COPYs are eliminated (and replaced with
  // get_local/set_local) in the ExplicitLocals pass.
  let isAsCheapAsAMove = 1, isCodeGenOnly = 1 in
  def COPY_#vt : I<(outs vt:$res), (ins vt:$src), [], "copy_local\t$res, $src">;

  // TEE is similar to COPY, but writes two copies of its result. Typically
  // this would be used to stackify one result and write the other result to a
  // local.
  let isAsCheapAsAMove = 1, isCodeGenOnly = 1 in
  def TEE_#vt : I<(outs vt:$res, vt:$also), (ins vt:$src), [],
                  "tee_local\t$res, $also, $src">;

  // This is the actual get_local instruction in wasm. These are made explicit
  // by the ExplicitLocals pass. It has mayLoad because it reads from a wasm
  // local, which is a side effect not otherwise modeled in LLVM.
  let mayLoad = 1, isAsCheapAsAMove = 1 in
  def GET_LOCAL_#vt : I<(outs vt:$res), (ins local_op:$local), [],
                        "get_local\t$res, $local", 0x20>;

  // This is the actual set_local instruction in wasm. These are made explicit
  // by the ExplicitLocals pass. It has mayStore because it writes to a wasm
  // local, which is a side effect not otherwise modeled in LLVM.
  let mayStore = 1, isAsCheapAsAMove = 1 in
  def SET_LOCAL_#vt : I<(outs), (ins local_op:$local, vt:$src), [],
                        "set_local\t$local, $src", 0x21>;

  // This is the actual tee_local instruction in wasm. TEEs are turned into
  // TEE_LOCALs by the ExplicitLocals pass. It has mayStore for the same reason
  // as SET_LOCAL.
  let mayStore = 1, isAsCheapAsAMove = 1 in
  def TEE_LOCAL_#vt : I<(outs vt:$res), (ins local_op:$local, vt:$src), [],
                         "tee_local\t$res, $local, $src", 0x22>;

  // Unused values must be dropped in some contexts.
  def DROP_#vt : I<(outs), (ins vt:$src), [],
                   "drop\t$src", 0x1a>;

  let mayLoad = 1 in
  def GET_GLOBAL_#vt : I<(outs vt:$res), (ins global_op:$local), [],
                         "get_global\t$res, $local", 0x23>;

  let mayStore = 1 in
  def SET_GLOBAL_#vt : I<(outs), (ins global_op:$local, vt:$src), [],
                         "set_global\t$local, $src", 0x24>;

} // hasSideEffects = 0
}
defm : LOCAL<I32>;
defm : LOCAL<I64>;
defm : LOCAL<F32>;
defm : LOCAL<F64>;
defm : LOCAL<V128>, Requires<[HasSIMD128]>;

let isMoveImm = 1, isAsCheapAsAMove = 1, isReMaterializable = 1 in {
def CONST_I32 : I<(outs I32:$res), (ins i32imm_op:$imm),
                  [(set I32:$res, imm:$imm)],
                  "i32.const\t$res, $imm", 0x41>;
def CONST_I64 : I<(outs I64:$res), (ins i64imm_op:$imm),
                  [(set I64:$res, imm:$imm)],
                  "i64.const\t$res, $imm", 0x42>;
def CONST_F32 : I<(outs F32:$res), (ins f32imm_op:$imm),
                  [(set F32:$res, fpimm:$imm)],
                  "f32.const\t$res, $imm", 0x43>;
def CONST_F64 : I<(outs F64:$res), (ins f64imm_op:$imm),
                  [(set F64:$res, fpimm:$imm)],
                  "f64.const\t$res, $imm", 0x44>;
} // isMoveImm = 1, isAsCheapAsAMove = 1, isReMaterializable = 1

} // Defs = [ARGUMENTS]

def : Pat<(i32 (WebAssemblywrapper tglobaladdr:$addr)),
          (CONST_I32 tglobaladdr:$addr)>;
def : Pat<(i32 (WebAssemblywrapper texternalsym:$addr)),
          (CONST_I32 texternalsym:$addr)>;

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
