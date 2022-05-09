//===-- EXPInstructions.td - Export Instruction Definitions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// EXP classes
//===----------------------------------------------------------------------===//

class EXPCommon<bit done, string asm = ""> : InstSI<
  (outs),
  (ins exp_tgt:$tgt,
       ExpSrc0:$src0, ExpSrc1:$src1, ExpSrc2:$src2, ExpSrc3:$src3,
       exp_vm:$vm, exp_compr:$compr, i32imm:$en),
  asm> {
  let EXP = 1;
  let EXP_CNT = 1;
  let mayLoad = done;
  let mayStore = 1;
  let UseNamedOperandTable = 1;
  let Uses = [EXEC];
  let SchedRW = [WriteExport];
  let DisableWQM = 1;
}

class EXP_Pseudo<bit done> : EXPCommon<done>,
                             SIMCInstr <NAME, SIEncodingFamily.NONE> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class EXP_Real<bit done, string pseudo, int subtarget>
  : EXPCommon<done, "exp$tgt $src0, $src1, $src2, $src3"#!if(done, " done", "")
                    #"$compr$vm">,
    SIMCInstr <pseudo, subtarget> {
  let AsmMatchConverter = "cvtExp";
}

//===----------------------------------------------------------------------===//
// EXP Instructions
//===----------------------------------------------------------------------===//

// Split EXP instruction into EXP and EXP_DONE so we can set
// mayLoad for done=1.
let SubtargetPredicate = isNotGFX90APlus in {
def EXP : EXP_Pseudo<0>;
def EXP_DONE : EXP_Pseudo<1>;
} // let SubtargetPredicate = isNotGFX90APlus

//===----------------------------------------------------------------------===//
// SI
//===----------------------------------------------------------------------===//

class EXP_Real_si<bit _done, string pseudo>
  : EXP_Real<_done, pseudo, SIEncodingFamily.SI>, EXPe {
  let AssemblerPredicate = isGFX6GFX7;
  let DecoderNamespace = "GFX6GFX7";
  let done = _done;
}

def EXP_si      : EXP_Real_si<0, "EXP">;
def EXP_DONE_si : EXP_Real_si<1, "EXP_DONE">;

//===----------------------------------------------------------------------===//
// VI
//===----------------------------------------------------------------------===//

class EXP_Real_vi<bit _done, string pseudo>
  : EXP_Real<_done, pseudo, SIEncodingFamily.VI>, EXPe_vi {
  let AssemblerPredicate = isGFX8GFX9;
  let SubtargetPredicate = isNotGFX90APlus;
  let DecoderNamespace = "GFX8";
  let done = _done;
}

def EXP_vi      : EXP_Real_vi<0, "EXP">;
def EXP_DONE_vi : EXP_Real_vi<1, "EXP_DONE">;

//===----------------------------------------------------------------------===//
// GFX10+
//===----------------------------------------------------------------------===//

class EXP_Real_gfx10<bit _done, string pseudo>
  : EXP_Real<_done, pseudo, SIEncodingFamily.GFX10>, EXPe {
  let AssemblerPredicate = isGFX10Plus;
  let DecoderNamespace = "GFX10";
  let done = _done;
}

def EXP_gfx10      : EXP_Real_gfx10<0, "EXP">;
def EXP_DONE_gfx10 : EXP_Real_gfx10<1, "EXP_DONE">;

//===----------------------------------------------------------------------===//
// EXP Patterns
//===----------------------------------------------------------------------===//

class ExpPattern<ValueType vt, Instruction Inst, int done_val> : GCNPat<
  (int_amdgcn_exp timm:$tgt, timm:$en,
                  (vt ExpSrc0:$src0), (vt ExpSrc1:$src1),
                  (vt ExpSrc2:$src2), (vt ExpSrc3:$src3),
                  done_val, timm:$vm),
  (Inst timm:$tgt, ExpSrc0:$src0, ExpSrc1:$src1,
        ExpSrc2:$src2, ExpSrc3:$src3, timm:$vm, 0, timm:$en)
>;

class ExpComprPattern<ValueType vt, Instruction Inst, int done_val> : GCNPat<
  (int_amdgcn_exp_compr timm:$tgt, timm:$en,
                        (vt ExpSrc0:$src0), (vt ExpSrc1:$src1),
                        done_val, timm:$vm),
  (Inst timm:$tgt, ExpSrc0:$src0, ExpSrc1:$src1,
        (IMPLICIT_DEF), (IMPLICIT_DEF), timm:$vm, 1, timm:$en)
>;

// FIXME: The generated DAG matcher seems to have strange behavior
// with a 1-bit literal to match, so use a -1 for checking a true
// 1-bit value.
def : ExpPattern<i32, EXP, 0>;
def : ExpPattern<i32, EXP_DONE, -1>;
def : ExpPattern<f32, EXP, 0>;
def : ExpPattern<f32, EXP_DONE, -1>;

def : ExpComprPattern<v2i16, EXP, 0>;
def : ExpComprPattern<v2i16, EXP_DONE, -1>;
def : ExpComprPattern<v2f16, EXP, 0>;
def : ExpComprPattern<v2f16, EXP_DONE, -1>;
