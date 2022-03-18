//===-- X86InstrCMovSetCC.td - Conditional Move and SetCC --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the X86 conditional move and set on condition
// instructions.
//
//===----------------------------------------------------------------------===//


// CMOV instructions.
let isCodeGenOnly = 1, ForceDisassemble = 1 in {
let Uses = [EFLAGS], Predicates = [HasCMOV], Constraints = "$src1 = $dst",
    isCommutable = 1, SchedRW = [WriteCMOV] in {
  def CMOV16rr
    : I<0x40, MRMSrcRegCC, (outs GR16:$dst), (ins GR16:$src1, GR16:$src2, ccode:$cond),
        "cmov${cond}{w}\t{$src2, $dst|$dst, $src2}",
        [(set GR16:$dst,
              (X86cmov GR16:$src1, GR16:$src2, timm:$cond, EFLAGS))]>,
              TB, OpSize16;
  def CMOV32rr
    : I<0x40, MRMSrcRegCC, (outs GR32:$dst), (ins GR32:$src1, GR32:$src2, ccode:$cond),
        "cmov${cond}{l}\t{$src2, $dst|$dst, $src2}",
        [(set GR32:$dst,
              (X86cmov GR32:$src1, GR32:$src2, timm:$cond, EFLAGS))]>,
              TB, OpSize32;
  def CMOV64rr
    :RI<0x40, MRMSrcRegCC, (outs GR64:$dst), (ins GR64:$src1, GR64:$src2, ccode:$cond),
        "cmov${cond}{q}\t{$src2, $dst|$dst, $src2}",
        [(set GR64:$dst,
              (X86cmov GR64:$src1, GR64:$src2, timm:$cond, EFLAGS))]>, TB;
}

let Uses = [EFLAGS], Predicates = [HasCMOV], Constraints = "$src1 = $dst",
    SchedRW = [WriteCMOV.Folded, WriteCMOV.ReadAfterFold] in {
  def CMOV16rm
    : I<0x40, MRMSrcMemCC, (outs GR16:$dst), (ins GR16:$src1, i16mem:$src2, ccode:$cond),
        "cmov${cond}{w}\t{$src2, $dst|$dst, $src2}",
        [(set GR16:$dst, (X86cmov GR16:$src1, (loadi16 addr:$src2),
                                  timm:$cond, EFLAGS))]>, TB, OpSize16;
  def CMOV32rm
    : I<0x40, MRMSrcMemCC, (outs GR32:$dst), (ins GR32:$src1, i32mem:$src2, ccode:$cond),
        "cmov${cond}{l}\t{$src2, $dst|$dst, $src2}",
        [(set GR32:$dst, (X86cmov GR32:$src1, (loadi32 addr:$src2),
                                  timm:$cond, EFLAGS))]>, TB, OpSize32;
  def CMOV64rm
    :RI<0x40, MRMSrcMemCC, (outs GR64:$dst), (ins GR64:$src1, i64mem:$src2, ccode:$cond),
        "cmov${cond}{q}\t{$src2, $dst|$dst, $src2}",
        [(set GR64:$dst, (X86cmov GR64:$src1, (loadi64 addr:$src2),
                                  timm:$cond, EFLAGS))]>, TB;
} // Uses = [EFLAGS], Predicates = [HasCMOV], Constraints = "$src1 = $dst"
} // isCodeGenOnly = 1, ForceDisassemble = 1

def inv_cond_XFORM : SDNodeXForm<imm, [{
  X86::CondCode CC = static_cast<X86::CondCode>(N->getZExtValue());
  return CurDAG->getTargetConstant(X86::GetOppositeBranchCondition(CC),
                                   SDLoc(N), MVT::i8);
}]>;

// Conditional moves with folded loads with operands swapped and conditions
// inverted.
let Predicates = [HasCMOV] in {
  def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, timm:$cond, EFLAGS),
            (CMOV16rm GR16:$src2, addr:$src1, (inv_cond_XFORM timm:$cond))>;
  def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, timm:$cond, EFLAGS),
            (CMOV32rm GR32:$src2, addr:$src1, (inv_cond_XFORM timm:$cond))>;
  def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, timm:$cond, EFLAGS),
            (CMOV64rm GR64:$src2, addr:$src1, (inv_cond_XFORM timm:$cond))>;
}

// SetCC instructions.
let Uses = [EFLAGS], isCodeGenOnly = 1, ForceDisassemble = 1 in {
  def SETCCr : I<0x90, MRMXrCC, (outs GR8:$dst), (ins ccode:$cond),
                "set${cond}\t$dst",
                [(set GR8:$dst, (X86setcc timm:$cond, EFLAGS))]>,
                TB, Sched<[WriteSETCC]>;
  def SETCCm : I<0x90, MRMXmCC, (outs), (ins i8mem:$dst, ccode:$cond),
                "set${cond}\t$dst",
                [(store (X86setcc timm:$cond, EFLAGS), addr:$dst)]>,
                TB, Sched<[WriteSETCCStore]>;
} // Uses = [EFLAGS]

multiclass CMOV_SETCC_Aliases<string Cond, int CC> {
  def : InstAlias<"cmov"#Cond#"{w}\t{$src, $dst|$dst, $src}",
                  (CMOV16rr GR16:$dst, GR16:$src, CC), 0>;
  def : InstAlias<"cmov"#Cond#"{w}\t{$src, $dst|$dst, $src}",
                  (CMOV16rm GR16:$dst, i16mem:$src, CC), 0>;
  def : InstAlias<"cmov"#Cond#"{l}\t{$src, $dst|$dst, $src}",
                  (CMOV32rr GR32:$dst, GR32:$src, CC), 0>;
  def : InstAlias<"cmov"#Cond#"{l}\t{$src, $dst|$dst, $src}",
                  (CMOV32rm GR32:$dst, i32mem:$src, CC), 0>;
  def : InstAlias<"cmov"#Cond#"{q}\t{$src, $dst|$dst, $src}",
                  (CMOV64rr GR64:$dst, GR64:$src, CC), 0>;
  def : InstAlias<"cmov"#Cond#"{q}\t{$src, $dst|$dst, $src}",
                  (CMOV64rm GR64:$dst, i64mem:$src, CC), 0>;

  def : InstAlias<"set"#Cond#"\t$dst", (SETCCr GR8:$dst, CC), 0>;
  def : InstAlias<"set"#Cond#"\t$dst", (SETCCm i8mem:$dst, CC), 0>;
}

defm : CMOV_SETCC_Aliases<"o" ,  0>;
defm : CMOV_SETCC_Aliases<"no",  1>;
defm : CMOV_SETCC_Aliases<"b" ,  2>;
defm : CMOV_SETCC_Aliases<"ae",  3>;
defm : CMOV_SETCC_Aliases<"e" ,  4>;
defm : CMOV_SETCC_Aliases<"ne",  5>;
defm : CMOV_SETCC_Aliases<"be",  6>;
defm : CMOV_SETCC_Aliases<"a" ,  7>;
defm : CMOV_SETCC_Aliases<"s" ,  8>;
defm : CMOV_SETCC_Aliases<"ns",  9>;
defm : CMOV_SETCC_Aliases<"p" , 10>;
defm : CMOV_SETCC_Aliases<"np", 11>;
defm : CMOV_SETCC_Aliases<"l" , 12>;
defm : CMOV_SETCC_Aliases<"ge", 13>;
defm : CMOV_SETCC_Aliases<"le", 14>;
defm : CMOV_SETCC_Aliases<"g" , 15>;

// SALC is an undocumented instruction. Information for this instruction can be found
// here http://www.rcollins.org/secrets/opcodes/SALC.html
// Set AL if carry. 
let Uses = [EFLAGS], Defs = [AL], SchedRW = [WriteALU] in {
  def SALC : I<0xD6, RawFrm, (outs), (ins), "salc", []>, Requires<[Not64BitMode]>;
}
