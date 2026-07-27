//===-- RISCVInstrInfoZilx.td - 'Zilx' instructions --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file describes the RISC-V instructions for 'Zilx' (Indexed Integer
/// Load).
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//

// Zilx indexed loads reuse the AMO major opcode. The addressing mode is
// selected by funct5 (inst[31:27]) and the access width/signedness reuses the
// base integer load funct3. rs1 is the index operand, rs2 is the base operand,
// and the aq and rl bits are always 0. The base register is written in
// parentheses to distinguish it from the index register.
let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class ZilxLoad<bits<5> funct5, bits<3> funct3, string opcodestr>
    : RVInstRAtomic<funct5, /*aq=*/0, /*rl=*/0, funct3, OPC_AMO,
                    (outs GPR:$rd), (ins GPRMemZeroOffset:$rs2, GPR:$rs1),
                    opcodestr, "$rd, $rs2, $rs1">;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZilx] in {
// Unscaled indexed loads: address = base + index. Byte loads are not provided
// in the unscaled mode.
def LXH  : ZilxLoad<0b10010, 0b001, "lxh">,
           Sched<[WriteLDH, ReadMemBase, ReadMemBase]>;
def LXW  : ZilxLoad<0b10010, 0b010, "lxw">,
           Sched<[WriteLDW, ReadMemBase, ReadMemBase]>;
def LXHU : ZilxLoad<0b10010, 0b101, "lxhu">,
           Sched<[WriteLDH, ReadMemBase, ReadMemBase]>;

// Scaled indexed loads: address = base + (index << log2(access-size)). Byte
// loads are provided only in scaled form (the byte scale factor is 1).
def LXSB  : ZilxLoad<0b11010, 0b000, "lxsb">,
            Sched<[WriteLDB, ReadMemBase, ReadMemBase]>;
def LXSH  : ZilxLoad<0b11010, 0b001, "lxsh">,
            Sched<[WriteLDH, ReadMemBase, ReadMemBase]>;
def LXSW  : ZilxLoad<0b11010, 0b010, "lxsw">,
            Sched<[WriteLDW, ReadMemBase, ReadMemBase]>;
def LXSBU : ZilxLoad<0b11010, 0b100, "lxsbu">,
            Sched<[WriteLDB, ReadMemBase, ReadMemBase]>;
def LXSHU : ZilxLoad<0b11010, 0b101, "lxshu">,
            Sched<[WriteLDH, ReadMemBase, ReadMemBase]>;

let append Predicates = [IsRV64] in {
// Unscaled indexed loads (RV64-only widths).
def LXD  : ZilxLoad<0b10010, 0b011, "lxd">,
           Sched<[WriteLDD, ReadMemBase, ReadMemBase]>;
def LXWU : ZilxLoad<0b10010, 0b110, "lxwu">,
           Sched<[WriteLDW, ReadMemBase, ReadMemBase]>;

// Scaled indexed loads (RV64-only widths).
def LXSD  : ZilxLoad<0b11010, 0b011, "lxsd">,
            Sched<[WriteLDD, ReadMemBase, ReadMemBase]>;
def LXSWU : ZilxLoad<0b11010, 0b110, "lxswu">,
            Sched<[WriteLDW, ReadMemBase, ReadMemBase]>;

// Scaled indexed loads with a zero-extended 32-bit index (RV64-only):
// address = base + (zext32(index) << log2(access-size)).
def LXSUWB  : ZilxLoad<0b11110, 0b000, "lxsuwb">,
              Sched<[WriteLDB, ReadMemBase, ReadMemBase]>;
def LXSUWH  : ZilxLoad<0b11110, 0b001, "lxsuwh">,
              Sched<[WriteLDH, ReadMemBase, ReadMemBase]>;
def LXSUWW  : ZilxLoad<0b11110, 0b010, "lxsuww">,
              Sched<[WriteLDW, ReadMemBase, ReadMemBase]>;
def LXSUWD  : ZilxLoad<0b11110, 0b011, "lxsuwd">,
              Sched<[WriteLDD, ReadMemBase, ReadMemBase]>;
def LXSUWBU : ZilxLoad<0b11110, 0b100, "lxsuwbu">,
              Sched<[WriteLDB, ReadMemBase, ReadMemBase]>;
def LXSUWHU : ZilxLoad<0b11110, 0b101, "lxsuwhu">,
              Sched<[WriteLDH, ReadMemBase, ReadMemBase]>;
def LXSUWWU : ZilxLoad<0b11110, 0b110, "lxsuwwu">,
              Sched<[WriteLDW, ReadMemBase, ReadMemBase]>;
} // append Predicates = [IsRV64]

} // Predicates = [HasStdExtZilx]

// `lxb`/`lxbu` are assembler pseudoinstructions for the scaled byte loads.
// They are parse-only aliases (EmitPriority 0) so the disassembler still prints
// the underlying `lxsb`/`lxsbu`.
let Predicates = [HasStdExtZilx] in {
def : InstAlias<"lxb $rd, $rs2, $rs1",  (LXSB  GPR:$rd, GPRMemZeroOffset:$rs2, GPR:$rs1), 0>;
def : InstAlias<"lxbu $rd, $rs2, $rs1", (LXSBU GPR:$rd, GPRMemZeroOffset:$rs2, GPR:$rs1), 0>;
}

//===----------------------------------------------------------------------===//
// Pattern matching
//===----------------------------------------------------------------------===//

// The complexities below are chosen to give the indexed loads a strict
// priority over the plain reg+imm loads and to encode which form is the more
// specific match. Without an explicit complexity these patterns tie with the
// base L{B,H,W,D}[U] patterns (which also match a "reg + reg" address as
// base+0) and, being listed later, would never be selected. The required
// ordering is:
//   base L*  <  scale-0 (unscaled) Zilx  <  scaled Zilx
// For a given load only the scale-0 form (which leaves the shift as a separate
// slli) and the exactly-matching scaled form can compete, so a scaled load
// only needs to outrank its own unscaled form; distinct non-zero scales never
// compete for the same node.
class AddrRegReg<int N>
    : ComplexPattern<iPTR, 2, "SelectAddrRegRegFixedScale<"#N#">",
                     [], [], !if(!eq(N, 0), 10, 11)>;
// The zero-extended-index forms additionally fold the zext, so they are the
// most specific match and must outrank the plain AddrRegReg forms above.
class AddrRegZextReg<int N>
    : ComplexPattern<i64, 2, "SelectAddrRegZextRegFixedScale<"#N#", 32>",
                     [], [], !if(!eq(N, 0), 20, 21)>;

def AddrRegReg0 : AddrRegReg<0>;
def AddrRegReg1 : AddrRegReg<1>;
def AddrRegReg2 : AddrRegReg<2>;
def AddrRegReg3 : AddrRegReg<3>;

def AddrRegZextReg0 : AddrRegZextReg<0>;
def AddrRegZextReg1 : AddrRegZextReg<1>;
def AddrRegZextReg2 : AddrRegZextReg<2>;
def AddrRegZextReg3 : AddrRegZextReg<3>;

// The addressing-mode ComplexPatterns bind the base first and the (shifted)
// index second. Zilx encodes rs2 as the base (written in parentheses) and rs1
// as the index, so the operands are emitted in that order.
class UnscaledIndexLoadPat<PatFrag LoadOp, RVInst Inst, ValueType vt = XLenVT>
    : Pat<(vt (LoadOp (AddrRegReg0 (XLenVT GPR:$base), (XLenVT GPR:$index)))),
          (Inst GPRMemZeroOffset:$base, GPR:$index)>;

class ScaledIndexLoadPat<PatFrag LoadOp, RVInst Inst, int Scale, ValueType vt = XLenVT>
    : Pat<(vt (LoadOp (!cast<AddrRegReg>("AddrRegReg" # Scale)
                      (XLenVT GPR:$base), (XLenVT GPR:$index)))),
          (Inst GPRMemZeroOffset:$base, GPR:$index)>;

class ScaledZextIndexLoadPat<PatFrag LoadOp, RVInst Inst, int Scale, ValueType vt = XLenVT>
    : Pat<(vt (LoadOp (!cast<AddrRegZextReg>("AddrRegZextReg" # Scale)
                      (XLenVT GPR:$base), (XLenVT GPR:$index)))),
          (Inst GPRMemZeroOffset:$base, GPR:$index)>;

// Patterns
let Predicates = [HasStdExtZilx] in {
// Byte loads only have a scaled (scale-0) encoding.
def : UnscaledIndexLoadPat<extloadi8, LXSB>;
def : UnscaledIndexLoadPat<sextloadi8, LXSB>;
def : UnscaledIndexLoadPat<zextloadi8, LXSBU>;

def : UnscaledIndexLoadPat<extloadi16, LXH>;
def : UnscaledIndexLoadPat<sextloadi16, LXH>;
def : UnscaledIndexLoadPat<zextloadi16, LXHU>;

def : UnscaledIndexLoadPat<load, LXW, i32>;

def : ScaledIndexLoadPat<extloadi16, LXSH, 1>;
def : ScaledIndexLoadPat<sextloadi16, LXSH, 1>;
def : ScaledIndexLoadPat<zextloadi16, LXSHU, 1>;

def : ScaledIndexLoadPat<load, LXSW, 2, i32>;

let append Predicates = [IsRV64] in {
def : UnscaledIndexLoadPat<extloadi32, LXW, i64>;
def : UnscaledIndexLoadPat<sextloadi32, LXW, i64>;
def : UnscaledIndexLoadPat<zextloadi32, LXWU, i64>;

def : UnscaledIndexLoadPat<load, LXD, i64>;

def : ScaledIndexLoadPat<extloadi32, LXSW, 2, i64>;
def : ScaledIndexLoadPat<sextloadi32, LXSW, 2, i64>;
def : ScaledIndexLoadPat<zextloadi32, LXSWU, 2, i64>;

def : ScaledIndexLoadPat<load, LXSD, 3, i64>;

// Zero-extended 32-bit index. Byte loads use the scale-0 encoding.
def : ScaledZextIndexLoadPat<extloadi8, LXSUWB, 0>;
def : ScaledZextIndexLoadPat<sextloadi8, LXSUWB, 0>;
def : ScaledZextIndexLoadPat<zextloadi8, LXSUWBU, 0>;

def : ScaledZextIndexLoadPat<extloadi16, LXSUWH, 1>;
def : ScaledZextIndexLoadPat<sextloadi16, LXSUWH, 1>;
def : ScaledZextIndexLoadPat<zextloadi16, LXSUWHU, 1>;

def : ScaledZextIndexLoadPat<load, LXSUWW, 2, i32>;

def : ScaledZextIndexLoadPat<extloadi32, LXSUWW, 2, i64>;
def : ScaledZextIndexLoadPat<sextloadi32, LXSUWW, 2, i64>;
def : ScaledZextIndexLoadPat<zextloadi32, LXSUWWU, 2, i64>;

def : ScaledZextIndexLoadPat<load, LXSUWD, 3, i64>;
} // append Predicates = [IsRV64]

} // Predicates = [HasStdExtZilx]
