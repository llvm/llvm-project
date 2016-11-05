//===- HexagonInstrInfoVector.td - Hexagon Vector Patterns -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon Vector instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

// Vector shift support. Vector shifting in Hexagon is rather different
// from internal representation of LLVM.
// LLVM assumes all shifts (in vector case) will have the form
// <VT> = SHL/SRA/SRL <VT> by <VT>
// while Hexagon has the following format:
// <VT> = SHL/SRA/SRL <VT> by <IT/i32>
// As a result, special care is needed to guarantee correctness and
// performance.
class vshift_v4i16<SDNode Op, string Str, bits<3>MajOp, bits<3>MinOp>
  : S_2OpInstImm<Str, MajOp, MinOp, u4_0Imm, []> {
  bits<4> src2;
  let Inst{11-8} = src2;
}

class vshift_v2i32<SDNode Op, string Str, bits<3>MajOp, bits<3>MinOp>
  : S_2OpInstImm<Str, MajOp, MinOp, u5_0Imm, []> {
  bits<5> src2;
  let Inst{12-8} = src2;
}

def S2_asr_i_vw : vshift_v2i32<sra, "vasrw", 0b010, 0b000>;
def S2_lsr_i_vw : vshift_v2i32<srl, "vlsrw", 0b010, 0b001>;
def S2_asl_i_vw : vshift_v2i32<shl, "vaslw", 0b010, 0b010>;

def S2_asr_i_vh : vshift_v4i16<sra, "vasrh", 0b100, 0b000>;
def S2_lsr_i_vh : vshift_v4i16<srl, "vlsrh", 0b100, 0b001>;
def S2_asl_i_vh : vshift_v4i16<shl, "vaslh", 0b100, 0b010>;

// Vector shift words by register
def S2_asr_r_vw : T_S3op_shiftVect < "vasrw", 0b00, 0b00>;
def S2_lsr_r_vw : T_S3op_shiftVect < "vlsrw", 0b00, 0b01>;
def S2_asl_r_vw : T_S3op_shiftVect < "vaslw", 0b00, 0b10>;
def S2_lsl_r_vw : T_S3op_shiftVect < "vlslw", 0b00, 0b11>;

// Vector shift halfwords by register
def S2_asr_r_vh : T_S3op_shiftVect < "vasrh", 0b01, 0b00>;
def S2_lsr_r_vh : T_S3op_shiftVect < "vlsrh", 0b01, 0b01>;
def S2_asl_r_vh : T_S3op_shiftVect < "vaslh", 0b01, 0b10>;
def S2_lsl_r_vh : T_S3op_shiftVect < "vlslh", 0b01, 0b11>;


// Hexagon doesn't have a vector multiply with C semantics.
// Instead, generate a pseudo instruction that gets expaneded into two
// scalar MPYI instructions.
// This is expanded by ExpandPostRAPseudos.
let isPseudo = 1 in
def PS_vmulw : PseudoM<(outs DoubleRegs:$Rd),
      (ins DoubleRegs:$Rs, DoubleRegs:$Rt), "", []>;

let isPseudo = 1 in
def PS_vmulw_acc : PseudoM<(outs DoubleRegs:$Rd),
      (ins DoubleRegs:$Rx, DoubleRegs:$Rs, DoubleRegs:$Rt), "", [],
      "$Rd = $Rx">;



