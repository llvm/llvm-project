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
multiclass Cmov<X86TypeInfo t, string args, bit ndd = 0, string suffix = ""> {
let isCommutable = 1, SchedRW = [WriteCMOV] in
  def rr#suffix : ITy<0x40, MRMSrcRegCC, t, (outs t.RegClass:$dst),
                      (ins t.RegClass:$src1, t.RegClass:$src2, ccode:$cond),
                      "cmov${cond}", args,
                      [(set t.RegClass:$dst, (X86cmov t.RegClass:$src1,
                                        t.RegClass:$src2, timm:$cond, EFLAGS))]>, UseEFLAGS, NDD<ndd>;
let SchedRW = [WriteCMOV.Folded, WriteCMOV.ReadAfterFold] in
  def rm#suffix : ITy<0x40, MRMSrcMemCC, t, (outs t.RegClass:$dst),
                      (ins t.RegClass:$src1, t.MemOperand:$src2, ccode:$cond),
                      "cmov${cond}", args,
                      [(set t.RegClass:$dst, (X86cmov t.RegClass:$src1,
                                    (t.LoadNode addr:$src2), timm:$cond, EFLAGS))]>, UseEFLAGS, NDD<ndd>;
}

multiclass Cfcmov<X86TypeInfo t> {
let isCommutable = 1, SchedRW = [WriteCMOV] in {
let Predicates = [HasCMOV, HasCF, In64BitMode] in {
  def rr : ITy<0x40, MRMDestRegCC, t, (outs t.RegClass:$dst),
               (ins t.RegClass:$src1, ccode:$cond),
               "cfcmov${cond}", unaryop_ndd_args,
               [(set t.RegClass:$dst,
                 (X86cmov 0, t.RegClass:$src1, timm:$cond, EFLAGS))]>, UseEFLAGS, NF;
  def rr_REV : ITy<0x40, MRMSrcRegCC, t, (outs t.RegClass:$dst),
                   (ins t.RegClass:$src1, ccode:$cond),
                   "cfcmov${cond}", unaryop_ndd_args,
                   []>, UseEFLAGS, EVEX, T_MAP4;
}
let Predicates = [HasCMOV, HasCF, HasNDD, In64BitMode] in
  def rr_ND : ITy<0x40, MRMSrcRegCC, t, (outs t.RegClass:$dst),
                  (ins t.RegClass:$src1, t.RegClass:$src2, ccode:$cond),
                  "cfcmov${cond}", binop_ndd_args, []>, UseEFLAGS, NDD<1>, NF;
}
let SchedRW = [WriteCMOV.Folded, WriteCMOV.ReadAfterFold] in {
  let Predicates = [HasCMOV, HasCF, In64BitMode], mayLoad = 1 in
    def rm : ITy<0x40, MRMSrcMemCC, t, (outs t.RegClass:$dst),
                 (ins t.MemOperand:$src1, ccode:$cond),
                 "cfcmov${cond}", unaryop_ndd_args, []>, UseEFLAGS, EVEX, T_MAP4;
  let Predicates = [HasCMOV, HasCF, HasNDD, In64BitMode], mayLoad = 1 in
    def rm_ND : ITy<0x40, MRMSrcMemCC, t, (outs t.RegClass:$dst),
                    (ins t.RegClass:$src1, t.MemOperand:$src2, ccode:$cond),
                    "cfcmov${cond}", binop_ndd_args, []>, UseEFLAGS, NDD<1>, NF;
}
let SchedRW = [WriteCMOV, ReadDefault, ReadDefault, ReadDefault, ReadDefault, ReadDefault],
    Predicates = [HasCMOV, HasCF, In64BitMode], mayStore = 1 in
  def mr : ITy<0x40, MRMDestMemCC, t, (outs),
                (ins t.MemOperand:$dst, t.RegClass:$src1, ccode:$cond),
                "cfcmov${cond}", unaryop_ndd_args, []>, UseEFLAGS, NF;
}

let isCodeGenOnly = 1, ForceDisassemble = 1 in {
  let Predicates = [HasCMOV, NoNDD], Constraints = "$dst = $src1" in {
    defm CMOV16 : Cmov<Xi16, binop_args>, OpSize16, TB;
    defm CMOV32 : Cmov<Xi32, binop_args>, OpSize32, TB;
    defm CMOV64 : Cmov<Xi64, binop_args>, TB;
  }

  let Predicates = [HasCMOV, HasNDD, In64BitMode] in {
    defm CMOV16 : Cmov<Xi16, binop_ndd_args, 1, "_ND">, PD;
    defm CMOV32 : Cmov<Xi32, binop_ndd_args, 1, "_ND">;
    defm CMOV64 : Cmov<Xi64, binop_ndd_args, 1, "_ND">;
  }

  defm CFCMOV16 : Cfcmov<Xi16>, PD;
  defm CFCMOV32 : Cfcmov<Xi32>;
  defm CFCMOV64 : Cfcmov<Xi64>;
} // isCodeGenOnly = 1, ForceDisassemble = 1

def inv_cond_XFORM : SDNodeXForm<imm, [{
  X86::CondCode CC = static_cast<X86::CondCode>(N->getZExtValue());
  return CurDAG->getTargetConstant(X86::GetOppositeBranchCondition(CC),
                                   SDLoc(N), MVT::i8);
}]>;

// Conditional moves with folded loads with operands swapped and conditions
// inverted.
let Predicates = [HasCMOV, NoNDD] in {
  def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, timm:$cond, EFLAGS),
            (CMOV16rm GR16:$src2, addr:$src1, (inv_cond_XFORM timm:$cond))>;
  def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, timm:$cond, EFLAGS),
            (CMOV32rm GR32:$src2, addr:$src1, (inv_cond_XFORM timm:$cond))>;
  def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, timm:$cond, EFLAGS),
            (CMOV64rm GR64:$src2, addr:$src1, (inv_cond_XFORM timm:$cond))>;
}

let Predicates = [HasCMOV, HasNDD] in {
  def : Pat<(X86cmov (loadi16 addr:$src1), GR16:$src2, timm:$cond, EFLAGS),
            (CMOV16rm_ND GR16:$src2, addr:$src1, (inv_cond_XFORM timm:$cond))>;
  def : Pat<(X86cmov (loadi32 addr:$src1), GR32:$src2, timm:$cond, EFLAGS),
            (CMOV32rm_ND GR32:$src2, addr:$src1, (inv_cond_XFORM timm:$cond))>;
  def : Pat<(X86cmov (loadi64 addr:$src1), GR64:$src2, timm:$cond, EFLAGS),
            (CMOV64rm_ND GR64:$src2, addr:$src1, (inv_cond_XFORM timm:$cond))>;
}

// Create pseudo instruction and do the pattern matching to them.
// We use a machine pass to lower these pseudos into cmov, in order
// to avoid backend optimizations
let Uses = [EFLAGS], isNotDuplicable = 1, isPseudo = 1 in {

  multiclass CTSELECT<X86TypeInfo t> {
    // register-only
    let isCommutable = 0, SchedRW = [WriteCMOV],
        AsmString = "ctselect\\t$dst, $src1, $src2, $cond" in {
      def rr : PseudoI<(outs t.RegClass:$dst),
                       (ins t.RegClass:$src1, t.RegClass:$src2, i8imm:$cond),
                       [(set t.RegClass:$dst, (X86ctselect t.RegClass:$src1, t.RegClass:$src2, timm:$cond, EFLAGS))]>;
    }

    // register-memory
    let SchedRW = [WriteCMOV.Folded, WriteCMOV.ReadAfterFold],
        AsmString = "ctselect\\t$dst, $src1, $src2, $cond" in {
      def rm : PseudoI<(outs t.RegClass:$dst),
                       (ins t.RegClass:$src1, t.MemOperand:$src2, i8imm:$cond),
                       [(set t.RegClass:$dst, (X86ctselect t.RegClass:$src1, (t.LoadNode addr:$src2), timm:$cond, EFLAGS))]>;
    }
  }
}

let isCodeGenOnly = 1, hasSideEffects = 1, ForceDisassemble = 1 in {
  let Predicates = [HasCMOV], Constraints = "$dst = $src1" in {
    defm CTSELECT16 : CTSELECT<Xi16>;
    defm CTSELECT32 : CTSELECT<Xi32>;
    defm CTSELECT64 : CTSELECT<Xi64>;
  }
}

// CTSELECT_VEC base class
class CTSELECT_VEC<RegisterClass VRc, RegisterClass GRc>
    : PseudoI<
        (outs VRc:$dst, VRc:$tmpx, GRc:$tmpg),
        (ins  VRc:$t,   VRc:$f,   i8imm:$cond),
        []
      > {
  let Uses            = [EFLAGS];
  let isPseudo        = 1;
  let isNotDuplicable = 1;
  let hasSideEffects  = 1;
  let AsmString       = "ctselect\t$dst, $f, $t, $cond";
  let SchedRW         = [];
}

// Width-specific class aliases
class CTSELECT_VEC128  : CTSELECT_VEC<VR128,  GR32>;
class CTSELECT_VEC128X : CTSELECT_VEC<VR128X, GR32>;
class CTSELECT_VEC256  : CTSELECT_VEC<VR256,  GR32>;
class CTSELECT_VEC512  : CTSELECT_VEC<VR512,  GR32>;


//===----------------------------------------------------------------------===//
// 128-bit pseudos (SSE2 baseline; we use PXOR/PAND/MOVD/PSHUFD in the expander)
//===----------------------------------------------------------------------===//
let Predicates = [HasSSE2] in {

  def CTSELECT_V4F32 : CTSELECT_VEC128 {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  def CTSELECT_V2F64 : CTSELECT_VEC128 {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  def CTSELECT_V4I32 : CTSELECT_VEC128 {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  def CTSELECT_V2I64 : CTSELECT_VEC128 {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  def CTSELECT_V8I16 : CTSELECT_VEC128 {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  def CTSELECT_V16I8 : CTSELECT_VEC128 {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  // If your build has v8f16, keep this; otherwise comment it out.
  def CTSELECT_V8F16 : CTSELECT_VEC128 {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
}

let Predicates = [HasAVX] in {

  def CTSELECT_V4F32X : CTSELECT_VEC128X {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  def CTSELECT_V2F64X : CTSELECT_VEC128X {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  def CTSELECT_V4I32X : CTSELECT_VEC128X {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  def CTSELECT_V2I64X : CTSELECT_VEC128X {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  def CTSELECT_V8I16X : CTSELECT_VEC128X {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  def CTSELECT_V16I8X : CTSELECT_VEC128X {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  // If your build has v8f16, keep this; otherwise comment it out.
  def CTSELECT_V8F16X : CTSELECT_VEC128X {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
}

//===----------------------------------------------------------------------===//
// 256-bit pseudos
//===----------------------------------------------------------------------===//
let Predicates = [HasAVX] in {

  def CTSELECT_V8F32  : CTSELECT_VEC256 {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  def CTSELECT_V4F64  : CTSELECT_VEC256 {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  def CTSELECT_V8I32  : CTSELECT_VEC256 {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  def CTSELECT_V4I64  : CTSELECT_VEC256 {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  def CTSELECT_V16I16 : CTSELECT_VEC256 {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  def CTSELECT_V32I8  : CTSELECT_VEC256 {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  // If your build has v16f16, keep this; otherwise comment it out.
  def CTSELECT_V16F16 : CTSELECT_VEC256 {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
}

//===----------------------------------------------------------------------===//
// 512-bit pseudos
//===----------------------------------------------------------------------===//

// Core AVX-512F types
let Predicates = [HasAVX512] in {
  def CTSELECT_V16F32 : CTSELECT_VEC512 {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  def CTSELECT_V8F64  : CTSELECT_VEC512 {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  def CTSELECT_V16I32 : CTSELECT_VEC512 {
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
  def CTSELECT_V8I64  : CTSELECT_VEC512{
    let Constraints = "@earlyclobber $dst,@earlyclobber $tmpx,@earlyclobber $tmpg";
  }
}

//===----------------------------------------------------------------------===//
// Selection patterns: X86ctselect(...), EFLAGS -> CTSELECT_V*
//
// NOTE:
//  * The SDNode carries Glue from CMP/TEST (due to SDNPInGlue).
//  * We list EFLAGS explicitly in the pattern (X86 style) to model the arch read.
//  * Temps (tmpx/tmpy,tmpg) are not in the pattern; they’re outs allocated by RA.
//===----------------------------------------------------------------------===//

let Predicates = [HasSSE2] in {

  // 128-bit integer
  def : Pat<(v4i32 (X86ctselect VR128:$t, VR128:$f, (i8 timm:$cc), EFLAGS)),
            (CTSELECT_V4I32 VR128:$t, VR128:$f, timm:$cc)>;
  def : Pat<(v2i64 (X86ctselect VR128:$t, VR128:$f, (i8 timm:$cc), EFLAGS)),
            (CTSELECT_V2I64 VR128:$t, VR128:$f, timm:$cc)>;
  def : Pat<(v8i16 (X86ctselect VR128:$t, VR128:$f, (i8 timm:$cc), EFLAGS)),
            (CTSELECT_V8I16 VR128:$t, VR128:$f, timm:$cc)>;
  def : Pat<(v16i8 (X86ctselect VR128:$t, VR128:$f, (i8 timm:$cc), EFLAGS)),
            (CTSELECT_V16I8 VR128:$t, VR128:$f, timm:$cc)>;

  // 128-bit float (bitwise-equivalent ops in expander)
  def : Pat<(v4f32 (X86ctselect VR128:$t, VR128:$f, (i8 timm:$cc), EFLAGS)),
            (CTSELECT_V4F32 VR128:$t, VR128:$f, timm:$cc)>;
  def : Pat<(v2f64 (X86ctselect VR128:$t, VR128:$f, (i8 timm:$cc), EFLAGS)),
            (CTSELECT_V2F64 VR128:$t, VR128:$f, timm:$cc)>;

  // 128-bit f16 (optional)
  def : Pat<(v8f16 (X86ctselect VR128:$t, VR128:$f, (i8 timm:$cc), EFLAGS)),
            (CTSELECT_V8F16 VR128:$t, VR128:$f, timm:$cc)>;
}

let Predicates = [HasAVX] in {

  // 256-bit integer
  def : Pat<(v8i32  (X86ctselect VR256:$t, VR256:$f, (i8 timm:$cc), EFLAGS)),
            (CTSELECT_V8I32  VR256:$t, VR256:$f, timm:$cc)>;
  def : Pat<(v4i64  (X86ctselect VR256:$t, VR256:$f, (i8 timm:$cc), EFLAGS)),
            (CTSELECT_V4I64  VR256:$t, VR256:$f, timm:$cc)>;
  def : Pat<(v16i16 (X86ctselect VR256:$t, VR256:$f, (i8 timm:$cc), EFLAGS)),
            (CTSELECT_V16I16 VR256:$t, VR256:$f, timm:$cc)>;
  def : Pat<(v32i8  (X86ctselect VR256:$t, VR256:$f, (i8 timm:$cc), EFLAGS)),
            (CTSELECT_V32I8  VR256:$t, VR256:$f, timm:$cc)>;

  // 256-bit float (bitwise-equivalent ops in expander)
  def : Pat<(v8f32 (X86ctselect VR256:$t, VR256:$f, (i8 timm:$cc), EFLAGS)),
            (CTSELECT_V8F32 VR256:$t, VR256:$f, timm:$cc)>;
  def : Pat<(v4f64 (X86ctselect VR256:$t, VR256:$f, (i8 timm:$cc), EFLAGS)),
            (CTSELECT_V4F64 VR256:$t, VR256:$f, timm:$cc)>;

  // 256-bit f16 (optional)
  def : Pat<(v16f16 (X86ctselect VR256:$t, VR256:$f, (i8 timm:$cc), EFLAGS)),
            (CTSELECT_V16F16 VR256:$t, VR256:$f, timm:$cc)>;
}

//===----------------------------------------------------------------------===//
// 512-bit selection patterns
//  Note: SDNode is X86ctselect with SDNPInGlue; pattern names EFLAGS explicitly.
//  Temps ($tmpx,$tmpg) are outs allocated by RA; not mentioned in patterns.
//===----------------------------------------------------------------------===//

let Predicates = [HasAVX512] in {
  def : Pat<(v16f32 (X86ctselect VR512:$t, VR512:$f, (i8 timm:$cc), EFLAGS)),
            (CTSELECT_V16F32 VR512:$t, VR512:$f, timm:$cc)>;
  def : Pat<(v8f64 (X86ctselect VR512:$t, VR512:$f, (i8 timm:$cc), EFLAGS)),
            (CTSELECT_V8F64  VR512:$t, VR512:$f, timm:$cc)>;

  def : Pat<(v16i32 (X86ctselect VR512:$t, VR512:$f, (i8 timm:$cc), EFLAGS)),
            (CTSELECT_V16I32 VR512:$t, VR512:$f, timm:$cc)>;
  def : Pat<(v8i64 (X86ctselect VR512:$t, VR512:$f, (i8 timm:$cc), EFLAGS)),
            (CTSELECT_V8I64  VR512:$t, VR512:$f, timm:$cc)>;
}

let Predicates = [HasCMOV, HasCF] in {
  def : Pat<(X86cmov GR16:$src1, 0, timm:$cond, EFLAGS),
            (CFCMOV16rr GR16:$src1, (inv_cond_XFORM timm:$cond))>;
  def : Pat<(X86cmov GR32:$src1, 0, timm:$cond, EFLAGS),
            (CFCMOV32rr GR32:$src1, (inv_cond_XFORM timm:$cond))>;
  def : Pat<(X86cmov GR64:$src1, 0, timm:$cond, EFLAGS),
            (CFCMOV64rr GR64:$src1, (inv_cond_XFORM timm:$cond))>;

  def : Pat<(X86cload addr:$src1, 0, timm:$cond, EFLAGS),
            (CFCMOV16rm addr:$src1, timm:$cond)>;
  def : Pat<(X86cload addr:$src1, 0, timm:$cond, EFLAGS),
            (CFCMOV32rm addr:$src1, timm:$cond)>;
  def : Pat<(X86cload addr:$src1, 0, timm:$cond, EFLAGS),
            (CFCMOV64rm addr:$src1, timm:$cond)>;

  def : Pat<(X86cload addr:$src2, GR16:$src1, timm:$cond, EFLAGS),
            (CFCMOV16rm_ND GR16:$src1, addr:$src2, timm:$cond)>;
  def : Pat<(X86cload addr:$src2, GR32:$src1, timm:$cond, EFLAGS),
            (CFCMOV32rm_ND GR32:$src1, addr:$src2, timm:$cond)>;
  def : Pat<(X86cload addr:$src2, GR64:$src1, timm:$cond, EFLAGS),
            (CFCMOV64rm_ND GR64:$src1, addr:$src2, timm:$cond)>;

  def : Pat<(X86cstore GR16:$src2, addr:$src1, timm:$cond, EFLAGS),
            (CFCMOV16mr addr:$src1, GR16:$src2, timm:$cond)>;
  def : Pat<(X86cstore GR32:$src2, addr:$src1, timm:$cond, EFLAGS),
            (CFCMOV32mr addr:$src1, GR32:$src2, timm:$cond)>;
  def : Pat<(X86cstore GR64:$src2, addr:$src1, timm:$cond, EFLAGS),
            (CFCMOV64mr addr:$src1, GR64:$src2, timm:$cond)>;
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

// SetZUCC and promoted SetCC instructions.
let Uses = [EFLAGS], isCodeGenOnly = 1, ForceDisassemble = 1,
  hasSideEffects = 0, Predicates = [In64BitMode], Predicates = [HasNDD] in {
  def SETZUCCr : I<0x40, MRMXrCC, (outs GR8:$dst), (ins ccode:$cond),
                "setzu${cond}\t$dst", []>,
                XD, ZU, NoCD8, Sched<[WriteSETCC]>;
  def SETCCr_EVEX : I<0x40, MRMXrCC, (outs GR8:$dst), (ins ccode:$cond),
                "set${cond}\t$dst", []>,
                XD, PL, Sched<[WriteSETCC]>;
  let mayStore = 1 in {
    def SETZUCCm : I<0x40, MRMXmCC, (outs), (ins i8mem:$dst, ccode:$cond),
                  "setzu${cond}\t$dst", []>,
                  XD, ZU, NoCD8, Sched<[WriteSETCCStore]>;
    def SETCCm_EVEX : I<0x40, MRMXmCC, (outs), (ins i8mem:$dst, ccode:$cond),
                  "set${cond}\t$dst", []>,
                  XD, PL, Sched<[WriteSETCCStore]>;
  }
}

// SALC is an undocumented instruction. Information for this instruction can be found
// here http://www.rcollins.org/secrets/opcodes/SALC.html
// Set AL if carry. 
let Uses = [EFLAGS], Defs = [AL], SchedRW = [WriteALU] in {
  def SALC : I<0xD6, RawFrm, (outs), (ins), "salc", []>, Requires<[Not64BitMode]>;
}
