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

class EXPCommon<dag outs, dag ins, string asm, list<dag> pattern> :
  InstSI<outs, ins, asm, pattern> {
  let EXP = 1;
  let EXP_CNT = 1;
  let mayLoad = 0; // Set to 1 if done bit is set.
  let mayStore = 1;
  let UseNamedOperandTable = 1;
  let Uses = [EXEC];
  let SchedRW = [WriteExport];
}

class EXP_Helper<bit done> : EXPCommon<
  (outs),
  (ins exp_tgt:$tgt,
       ExpSrc0:$src0, ExpSrc1:$src1, ExpSrc2:$src2, ExpSrc3:$src3,
       exp_vm:$vm, exp_compr:$compr, i32imm:$en),
  "exp$tgt $src0, $src1, $src2, $src3"#!if(done, " done", "")#"$compr$vm", []> {
  let AsmMatchConverter = "cvtExp";
}

// Split EXP instruction into EXP and EXP_DONE so we can set
// mayLoad for done=1.
multiclass EXP_m<bit done> {
  let mayLoad = done, DisableWQM = 1 in {
    let isPseudo = 1, isCodeGenOnly = 1 in {
      def "" : EXP_Helper<done>,
               SIMCInstr <NAME, SIEncodingFamily.NONE>;
    }

    let done = done in {
      def _si : EXP_Helper<done>,
                SIMCInstr <NAME, SIEncodingFamily.SI>,
                EXPe {
        let AssemblerPredicate = isGFX6GFX7;
        let DecoderNamespace = "GFX6GFX7";
      }

      def _vi : EXP_Helper<done>,
                SIMCInstr <NAME, SIEncodingFamily.VI>,
                EXPe_vi {
        let AssemblerPredicate = isGFX8GFX9;
        let DecoderNamespace = "GFX8";
      }

      def _gfx10 : EXP_Helper<done>,
                   SIMCInstr <NAME, SIEncodingFamily.GFX10>,
                   EXPe {
        let AssemblerPredicate = isGFX10Plus;
        let DecoderNamespace = "GFX10";
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// EXP Instructions
//===----------------------------------------------------------------------===//

defm EXP : EXP_m<0>;
defm EXP_DONE : EXP_m<1>;

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
