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

def SDT_WebAssemblyCallSeqStart : SDCallSeqStart<[SDTCisVT<0, iPTR>]>;
def SDT_WebAssemblyCallSeqEnd :
    SDCallSeqEnd<[SDTCisVT<0, iPTR>, SDTCisVT<1, iPTR>]>;
def SDT_WebAssemblyCall0    : SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>;
def SDT_WebAssemblyCall1    : SDTypeProfile<1, -1, [SDTCisPtrTy<1>]>;
def SDT_WebAssemblySwitch   : SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>;
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
def WebAssemblyswitch : SDNode<"WebAssemblyISD::SWITCH",
                               SDT_WebAssemblySwitch,
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

/*
 * TODO(jfb): Add the following.
 *
 * get_local: read the current value of a local variable
 * set_local: set the current value of a local variable
*/

def bb_op : Operand<OtherVT>;
def tjumptable_op : Operand<iPTR>;
def global : Operand<iPTR>;

//===----------------------------------------------------------------------===//
// WebAssembly Instruction Format Definitions.
//===----------------------------------------------------------------------===//

include "WebAssemblyInstrFormats.td"

multiclass ARGUMENT<WebAssemblyRegClass vt> {
  def ARGUMENT_#vt : I<(outs vt:$res), (ins i32imm:$argno),
                       [(set vt:$res, (WebAssemblyargument timm:$argno))]>;
}
defm : ARGUMENT<I32>;
defm : ARGUMENT<I64>;
defm : ARGUMENT<F32>;
defm : ARGUMENT<F64>;


def Const_I32 : I<(outs I32:$res), (ins i32imm:$imm),
                  [(set I32:$res, imm:$imm)],
                  "i32.const $res, $imm">;
def Const_I64 : I<(outs I64:$res), (ins i64imm:$imm),
                  [(set I64:$res, imm:$imm)],
                  "i64.const $res, $imm">;
def Const_F32 : I<(outs F32:$res), (ins f32imm:$imm),
                  [(set F32:$res, fpimm:$imm)],
                  "f32.const $res, $imm">;
def Const_F64 : I<(outs F64:$res), (ins f64imm:$imm),
                  [(set F64:$res, fpimm:$imm)],
                  "f64.const $res, $imm">;

// Special types of immediates. FIXME: Hard-coded as 32-bit for now.
def GLOBAL : I<(outs I32:$dst), (ins global:$addr),
               [(set I32:$dst, (WebAssemblywrapper tglobaladdr:$addr))],
               "global $dst, $addr">;
def JUMP_TABLE : I<(outs I32:$dst), (ins tjumptable_op:$addr),
                   [(set I32:$dst, (WebAssemblywrapper tjumptable:$addr))],
                   "jump_table $dst, $addr">;

//===----------------------------------------------------------------------===//
// Additional sets of instructions.
//===----------------------------------------------------------------------===//

include "WebAssemblyInstrMemory.td"
include "WebAssemblyInstrCall.td"
include "WebAssemblyInstrControl.td"
include "WebAssemblyInstrInteger.td"
include "WebAssemblyInstrFloat.td"
include "WebAssemblyInstrConv.td"
include "WebAssemblyInstrAtomics.td"
include "WebAssemblyInstrSIMD.td"
