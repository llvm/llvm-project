//=- HexagonInstrInfoV3.td - Target Desc. for Hexagon Target -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon V3 instructions in TableGen format.
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// J +
//===----------------------------------------------------------------------===//
// Call subroutine.
let isCall = 1, neverHasSideEffects = 1,
  Defs = [D0, D1, D2, D3, D4, D5, D6, D7, R28, R31,
                P0, P1, P2, P3, LC0, LC1, SA0, SA1] in {
  def CALLv3 : JInst<(outs), (ins calltarget:$dst),
             "call $dst", []>, Requires<[HasV3T]>;
}

//===----------------------------------------------------------------------===//
// J -
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// JR +
//===----------------------------------------------------------------------===//
// Call subroutine from register.
let isCall = 1, neverHasSideEffects = 1,
  Defs = [D0, D1, D2, D3, D4, D5, D6, D7, R28, R31,
                P0, P1, P2, P3, LC0, LC1, SA0, SA1] in {
  def CALLRv3 : JRInst<(outs), (ins IntRegs:$dst),
              "callr $dst",
              []>, Requires<[HasV3TOnly]>;
 }


// Jump to address from register
// if(p?.new) jumpr:t r?
let isReturn = 1, isTerminator = 1, isBarrier = 1,
  Defs = [PC], Uses = [R31] in {
  def JMPR_cdnPt_V3: JRInst<(outs), (ins PredRegs:$src1, IntRegs:$src2),
                       "if ($src1.new) jumpr:t $src2",
                       []>, Requires<[HasV3T]>;
}

// if (!p?.new) jumpr:t r?
let isReturn = 1, isTerminator = 1, isBarrier = 1,
  Defs = [PC], Uses = [R31] in {
  def JMPR_cdnNotPt_V3: JRInst<(outs), (ins PredRegs:$src1, IntRegs:$src2),
                       "if (!$src1.new) jumpr:t $src2",
                       []>, Requires<[HasV3T]>;
}

// Not taken.
// if(p?.new) jumpr:nt r?
let isReturn = 1, isTerminator = 1, isBarrier = 1,
  Defs = [PC], Uses = [R31] in {
  def JMPR_cdnPnt: JRInst<(outs), (ins PredRegs:$src1, IntRegs:$src2),
                       "if ($src1.new) jumpr:nt $src2",
                       []>, Requires<[HasV3T]>;
}

// if (!p?.new) jumpr:nt r?
let isReturn = 1, isTerminator = 1, isBarrier = 1,
  Defs = [PC], Uses = [R31] in {
  def JMPR_cdnNotPnt: JRInst<(outs), (ins PredRegs:$src1, IntRegs:$src2),
                       "if (!$src1.new) jumpr:nt $src2",
                       []>, Requires<[HasV3T]>;
}

//===----------------------------------------------------------------------===//
// JR -
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ALU64/ALU +
//===----------------------------------------------------------------------===//

let AddedComplexity = 200 in
def MAXw_dd : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                    DoubleRegs:$src2),
              "$dst = max($src2, $src1)",
              [(set (i64 DoubleRegs:$dst),
                    (i64 (select (i1 (setlt (i64 DoubleRegs:$src2),
                                            (i64 DoubleRegs:$src1))),
                                 (i64 DoubleRegs:$src1),
                                 (i64 DoubleRegs:$src2))))]>,
Requires<[HasV3T]>;

let AddedComplexity = 200 in
def MINw_dd : ALU64_rr<(outs DoubleRegs:$dst), (ins DoubleRegs:$src1,
                                                    DoubleRegs:$src2),
              "$dst = min($src2, $src1)",
              [(set (i64 DoubleRegs:$dst),
                    (i64 (select (i1 (setgt (i64 DoubleRegs:$src2),
                                            (i64 DoubleRegs:$src1))),
                                 (i64 DoubleRegs:$src1),
                                 (i64 DoubleRegs:$src2))))]>,
Requires<[HasV3T]>;

//===----------------------------------------------------------------------===//
// ALU64/ALU -
//===----------------------------------------------------------------------===//




//def : Pat <(brcond (i1 (seteq (i32 IntRegs:$src1), 0)), bb:$offset),
//      (JMP_RegEzt (i32 IntRegs:$src1), bb:$offset)>, Requires<[HasV3T]>;

//def : Pat <(brcond (i1 (setne (i32 IntRegs:$src1), 0)), bb:$offset),
//      (JMP_RegNzt (i32 IntRegs:$src1), bb:$offset)>, Requires<[HasV3T]>;

//def : Pat <(brcond (i1 (setle (i32 IntRegs:$src1), 0)), bb:$offset),
//      (JMP_RegLezt (i32 IntRegs:$src1), bb:$offset)>, Requires<[HasV3T]>;

//def : Pat <(brcond (i1 (setge (i32 IntRegs:$src1), 0)), bb:$offset),
//      (JMP_RegGezt (i32 IntRegs:$src1), bb:$offset)>, Requires<[HasV3T]>;

//def : Pat <(brcond (i1 (setgt (i32 IntRegs:$src1), -1)), bb:$offset),
//      (JMP_RegGezt (i32 IntRegs:$src1), bb:$offset)>, Requires<[HasV3T]>;


// Map call instruction
def : Pat<(call (i32 IntRegs:$dst)),
      (CALLRv3 (i32 IntRegs:$dst))>, Requires<[HasV3T]>;
def : Pat<(call tglobaladdr:$dst),
      (CALLv3 tglobaladdr:$dst)>, Requires<[HasV3T]>;
def : Pat<(call texternalsym:$dst),
      (CALLv3 texternalsym:$dst)>, Requires<[HasV3T]>;
