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
    let isCommutable = 0, SchedRW = [WriteCMOV] in {
      def rr : PseudoI<(outs t.RegClass:$dst),
                       (ins t.RegClass:$src1, t.RegClass:$src2, i8imm:$cond),
                       [(set t.RegClass:$dst, (X86ctselect t.RegClass:$src1, t.RegClass:$src2, timm:$cond, EFLAGS))]>;
    }

    // register-memory
    let SchedRW = [WriteCMOV.Folded, WriteCMOV.ReadAfterFold] in {
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

let Uses = [EFLAGS], usesCustomInserter = 1, isNotDuplicable = 1, isPseudo = 1, hasSideEffects = 1 in {
  // 128-bit vector types
  let Predicates = [HasSSE1] in {
    def CTSELECT_V4F32  : PseudoI<(outs VR128:$dst), (ins VR128:$t, VR128:$f, i8imm:$cond),
                                  [(set VR128:$dst, (v4f32 (X86ctselect VR128:$t, VR128:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
    def CTSELECT_V2F64  : PseudoI<(outs VR128:$dst), (ins VR128:$t, VR128:$f, i8imm:$cond),
                                  [(set VR128:$dst, (v2f64 (X86ctselect VR128:$t, VR128:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
    def CTSELECT_V4I32  : PseudoI<(outs VR128:$dst), (ins VR128:$t, VR128:$f, i8imm:$cond),
                                  [(set VR128:$dst, (v4i32 (X86ctselect VR128:$t, VR128:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
    def CTSELECT_V2I64  : PseudoI<(outs VR128:$dst), (ins VR128:$t, VR128:$f, i8imm:$cond),
                                  [(set VR128:$dst, (v2i64 (X86ctselect VR128:$t, VR128:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
    def CTSELECT_V8I16  : PseudoI<(outs VR128:$dst), (ins VR128:$t, VR128:$f, i8imm:$cond),
                                  [(set VR128:$dst, (v8i16 (X86ctselect VR128:$t, VR128:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
    def CTSELECT_V16I8  : PseudoI<(outs VR128:$dst), (ins VR128:$t, VR128:$f, i8imm:$cond),
                                  [(set VR128:$dst, (v16i8 (X86ctselect VR128:$t, VR128:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
    def CTSELECT_V8F16  : PseudoI<(outs VR128:$dst), (ins VR128:$t, VR128:$f, i8imm:$cond),
                                  [(set VR128:$dst, (v8f16 (X86ctselect VR128:$t, VR128:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
  }

  // 128-bit vector types (AVX versions)
  let Predicates = [HasAVX] in {
    def CTSELECT_V4F32X : PseudoI<(outs VR128X:$dst), (ins VR128X:$t, VR128X:$f, i8imm:$cond),
                                  [(set VR128X:$dst, (v4f32 (X86ctselect VR128X:$t, VR128X:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
    def CTSELECT_V2F64X : PseudoI<(outs VR128X:$dst), (ins VR128X:$t, VR128X:$f, i8imm:$cond),
                                  [(set VR128X:$dst, (v2f64 (X86ctselect VR128X:$t, VR128X:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
    def CTSELECT_V4I32X : PseudoI<(outs VR128X:$dst), (ins VR128X:$t, VR128X:$f, i8imm:$cond),
                                  [(set VR128X:$dst, (v4i32 (X86ctselect VR128X:$t, VR128X:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
    def CTSELECT_V2I64X : PseudoI<(outs VR128X:$dst), (ins VR128X:$t, VR128X:$f, i8imm:$cond),
                                  [(set VR128X:$dst, (v2i64 (X86ctselect VR128X:$t, VR128X:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
    def CTSELECT_V8I16X : PseudoI<(outs VR128X:$dst), (ins VR128X:$t, VR128X:$f, i8imm:$cond),
                                  [(set VR128X:$dst, (v8i16 (X86ctselect VR128X:$t, VR128X:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
    def CTSELECT_V16I8X : PseudoI<(outs VR128X:$dst), (ins VR128X:$t, VR128X:$f, i8imm:$cond),
                                  [(set VR128X:$dst, (v16i8 (X86ctselect VR128X:$t, VR128X:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
    def CTSELECT_V8F16X : PseudoI<(outs VR128X:$dst), (ins VR128X:$t, VR128X:$f, i8imm:$cond),
                                  [(set VR128X:$dst, (v8f16 (X86ctselect VR128X:$t, VR128X:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
  }

  // 256-bit vector types
  let Predicates = [HasAVX] in {
    def CTSELECT_V8F32  : PseudoI<(outs VR256:$dst), (ins VR256:$t, VR256:$f, i8imm:$cond),
                                  [(set VR256:$dst, (v8f32 (X86ctselect VR256:$t, VR256:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
    def CTSELECT_V4F64  : PseudoI<(outs VR256:$dst), (ins VR256:$t, VR256:$f, i8imm:$cond),
                                  [(set VR256:$dst, (v4f64 (X86ctselect VR256:$t, VR256:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
    def CTSELECT_V8I32  : PseudoI<(outs VR256:$dst), (ins VR256:$t, VR256:$f, i8imm:$cond),
                                  [(set VR256:$dst, (v8i32 (X86ctselect VR256:$t, VR256:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
    def CTSELECT_V4I64  : PseudoI<(outs VR256:$dst), (ins VR256:$t, VR256:$f, i8imm:$cond),
                                  [(set VR256:$dst, (v4i64 (X86ctselect VR256:$t, VR256:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
    def CTSELECT_V16I16 : PseudoI<(outs VR256:$dst), (ins VR256:$t, VR256:$f, i8imm:$cond),
                                  [(set VR256:$dst, (v16i16 (X86ctselect VR256:$t, VR256:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
    def CTSELECT_V32I8  : PseudoI<(outs VR256:$dst), (ins VR256:$t, VR256:$f, i8imm:$cond),
                                  [(set VR256:$dst, (v32i8 (X86ctselect VR256:$t, VR256:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
    def CTSELECT_V16F16 : PseudoI<(outs VR256:$dst), (ins VR256:$t, VR256:$f, i8imm:$cond),
                                  [(set VR256:$dst, (v16f16 (X86ctselect VR256:$t, VR256:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[]>;
  }

  // 512-bit vector types
  let Predicates = [HasAVX512] in {
    def CTSELECT_V16F32 : PseudoI<(outs VR512:$dst), (ins VR512:$t, VR512:$f, i8imm:$cond),
                                  [(set VR512:$dst, (v16f32 (X86ctselect VR512:$t, VR512:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[WriteCMOV]>;
    def CTSELECT_V8F64  : PseudoI<(outs VR512:$dst), (ins VR512:$t, VR512:$f, i8imm:$cond),
                                  [(set VR512:$dst, (v8f64 (X86ctselect VR512:$t, VR512:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[WriteCMOV]>;
    def CTSELECT_V16I32 : PseudoI<(outs VR512:$dst), (ins VR512:$t, VR512:$f, i8imm:$cond),
                                  [(set VR512:$dst, (v16i32 (X86ctselect VR512:$t, VR512:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[WriteCMOV]>;
    def CTSELECT_V8I64  : PseudoI<(outs VR512:$dst), (ins VR512:$t, VR512:$f, i8imm:$cond),
                                  [(set VR512:$dst, (v8i64 (X86ctselect VR512:$t, VR512:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[WriteCMOV]>;
    def CTSELECT_V32I16 : PseudoI<(outs VR512:$dst), (ins VR512:$t, VR512:$f, i8imm:$cond),
                                  [(set VR512:$dst, (v32i16 (X86ctselect VR512:$t, VR512:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[WriteCMOV]>;
    def CTSELECT_V64I8  : PseudoI<(outs VR512:$dst), (ins VR512:$t, VR512:$f, i8imm:$cond),
                                  [(set VR512:$dst, (v64i8 (X86ctselect VR512:$t, VR512:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[WriteCMOV]>;
    def CTSELECT_V32F16 : PseudoI<(outs VR512:$dst), (ins VR512:$t, VR512:$f, i8imm:$cond),
                                  [(set VR512:$dst, (v32f16 (X86ctselect VR512:$t, VR512:$f, timm:$cond, EFLAGS)))]>,
                                  Sched<[WriteCMOV]>;
  }
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
