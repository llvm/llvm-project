; RUN: not --crash llc -O0 -global-isel -global-isel-abort=1 -verify-machineinstrs %s -o - 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: llc -O0 -global-isel -global-isel-abort=0 -verify-machineinstrs %s -o - 2>&1 | FileCheck %s --check-prefix=FALLBACK
; RUN: llc -O0 -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' -verify-machineinstrs %s -o %t.out 2> %t.err
; RUN: FileCheck %s --check-prefix=FALLBACK-WITH-REPORT-OUT < %t.out
; RUN: FileCheck %s --check-prefix=FALLBACK-WITH-REPORT-ERR < %t.err
; RUN: not --crash llc -global-isel -mtriple aarch64_be %s -o - 2>&1 | FileCheck %s --check-prefix=BIG-ENDIAN
; This file checks that the fallback path to selection dag works.
; The test is fragile in the sense that it must be updated to expose
; something that fails with global-isel.
; When we cannot produce a test case anymore, that means we can remove
; the fallback path.

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--"

; BIG-ENDIAN: unable to translate in big endian mode

; We use __fixunstfti as the common denominator for __fixunstfti on Linux and
; ___fixunstfti on iOS
; ERROR: unable to translate instruction: ret
; FALLBACK: ldr q0,
; FALLBACK-NEXT: bl __fixunstfti
;
; FALLBACK-WITH-REPORT-ERR: unable to translate instruction: ret
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for ABIi128
; FALLBACK-WITH-REPORT-OUT-LABEL: ABIi128:
; FALLBACK-WITH-REPORT-OUT: ldr q0,
; FALLBACK-WITH-REPORT-OUT-NEXT: bl __fixunstfti
define i128 @ABIi128(i128 %arg1) {
  %farg1 =       bitcast i128 %arg1 to fp128
  %res = fptoui fp128 %farg1 to i128
  ret i128 %res
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: G_STORE %3:_(<3 x s32>), %4:_(p0) :: (store 12 into %ir.addr + 16, align 16, basealign 32) (in function: odd_vector)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for odd_vector
; FALLBACK-WITH-REPORT-OUT-LABEL: odd_vector:
define void @odd_vector(<7 x i32>* %addr) {
  %vec = load <7 x i32>, <7 x i32>* %addr
  store <7 x i32> %vec, <7 x i32>* %addr
  ret void
}

  ; AArch64 was asserting instead of returning an invalid mapping for unknown
  ; sizes.
; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction: ret: '  ret i128 undef' (in function: sequence_sizes)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for sequence_sizes
; FALLBACK-WITH-REPORT-LABEL: sequence_sizes:
define i128 @sequence_sizes([8 x i8] %in) {
  ret i128 undef
}

; Make sure we don't mess up metadata arguments.
declare void @llvm.write_register.i64(metadata, i64)

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: G_WRITE_REGISTER !0, %0:_(s64) (in function: test_write_register_intrin)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for test_write_register_intrin
; FALLBACK-WITH-REPORT-LABEL: test_write_register_intrin:
define void @test_write_register_intrin() {
  call void @llvm.write_register.i64(metadata !{!"sp"}, i64 0)
  ret void
}

@_ZTIi = external global i8*
declare i32 @__gxx_personality_v0(...)

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: %2:_(<2 x p0>) = G_INSERT_VECTOR_ELT %0:_, %{{[0-9]+}}:_(p0), %{{[0-9]+}}:_(s32) (in function: vector_of_pointers_insertelement)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for vector_of_pointers_insertelement
; FALLBACK-WITH-REPORT-OUT-LABEL: vector_of_pointers_insertelement:
define void @vector_of_pointers_insertelement() {
  br label %end

block:
  %dummy = insertelement <2 x i16*> %vec, i16* null, i32 0
  store <2 x i16*> %dummy, <2 x i16*>* undef
  ret void

end:
  %vec = load <2 x i16*>, <2 x i16*>* undef
  br label %block
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: %{{[0-9]+}}:_(s96) = G_ADD %{{[0-9]+}}:_, %{{[0-9]+}}:_ (in function: nonpow2_add_narrowing)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for nonpow2_add_narrowing
; FALLBACK-WITH-REPORT-OUT-LABEL: nonpow2_add_narrowing:
define void @nonpow2_add_narrowing(i128 %x, i128 %y) {
  %a = add i128 %x, %y
  %b = trunc i128 %a to i96
  %dummy = add i96 %b, %b
  store i96 %dummy, i96* undef
  ret void
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: %{{[0-9]+}}:_(s96) = G_INSERT %{{[0-9]+}}:_, %{{[0-9]+}}:_(s32), 64 (in function: nonpow2_or_narrowing)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for nonpow2_or_narrowing
; FALLBACK-WITH-REPORT-OUT-LABEL: nonpow2_or_narrowing:
define void @nonpow2_or_narrowing() {
  %a = add i128 undef, undef
  %b = trunc i128 %a to i96
  %a2 = add i128 undef, undef
  %b2 = trunc i128 %a2 to i96
  %dummy = or i96 %b, %b2
  store i96 %dummy, i96* undef
  ret void
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: %0:_(s96) = G_INSERT %10:_, %8:_(s32), 64 (in function: nonpow2_load_narrowing)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for nonpow2_load_narrowing
; FALLBACK-WITH-REPORT-OUT-LABEL: nonpow2_load_narrowing:
define void @nonpow2_load_narrowing() {
  %dummy = load i96, i96* undef
  store i96 %dummy, i96* undef
  ret void
}

; Currently can't handle vector lengths that aren't an exact multiple of
; natively supported vector lengths. Test that the fall-back works for those.
; FALLBACK-WITH-REPORT-ERR-G_IMPLICIT_DEF-LEGALIZABLE: (FIXME: this is what is expected once we can legalize non-pow-of-2 G_IMPLICIT_DEF) remark: <unknown>:0:0: unable to legalize instruction: %1:_(<7 x s64>) = G_ADD %0, %0 (in function: nonpow2_vector_add_fewerelements
; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: %{{[0-9]+}}:_(s64) = G_EXTRACT_VECTOR_ELT %{{[0-9]+}}:_(<7 x s64>), %{{[0-9]+}}:_(s64) (in function: nonpow2_vector_add_fewerelements)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for nonpow2_vector_add_fewerelements
; FALLBACK-WITH-REPORT-OUT-LABEL: nonpow2_vector_add_fewerelements:
define void @nonpow2_vector_add_fewerelements() {
  %dummy = add <7 x i64> undef, undef
  %ex = extractelement <7 x i64> %dummy, i64 0
  store i64 %ex, i64* undef
  ret void
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: cannot select: RET_ReallyLR implicit $x0 (in function: strict_align_feature)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for strict_align_feature
; FALLBACK-WITH-REPORT-OUT-LABEL: strict_align_feature
define i64 @strict_align_feature(i64* %p) #0 {
  %x = load i64, i64* %p, align 1
  ret i64 %x
}

attributes #0 = { "target-features"="+strict-align" }

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction: call
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for direct_mem
; FALLBACK-WITH-REPORT-OUT-LABEL: direct_mem
define void @direct_mem(i32 %x, i32 %y) {
entry:
  tail call void asm sideeffect "", "imr,imr,~{memory}"(i32 %x, i32 %y)
  ret void
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to lower function{{.*}}scalable_arg
; FALLBACK-WITH-REPORT-OUT-LABEL: scalable_arg
define <vscale x 16 x i8> @scalable_arg(<vscale x 16 x i1> %pred, i8* %addr) #1 {
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1.nxv16i8(<vscale x 16 x i1> %pred, i8* %addr)
  ret <vscale x 16 x i8> %res
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to lower function{{.*}}scalable_ret
; FALLBACK-WITH-REPORT-OUT-LABEL: scalable_ret
define <vscale x 16 x i8> @scalable_ret(i8* %addr) #1 {
  %pred = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 0)
  %res = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1.nxv16i8(<vscale x 16 x i1> %pred, i8* %addr)
  ret <vscale x 16 x i8> %res
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction{{.*}}scalable_call
; FALLBACK-WITH-REPORT-OUT-LABEL: scalable_call
define i8 @scalable_call(i8* %addr) #1 {
  %pred = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 0)
  %vec = call <vscale x 16 x i8> @llvm.aarch64.sve.ld1.nxv16i8(<vscale x 16 x i1> %pred, i8* %addr)
  %res = extractelement <vscale x 16 x i8> %vec, i32 0
  ret i8 %res
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction{{.*}}scalable_alloca
; FALLBACK-WITH-REPORT-OUT-LABEL: scalable_alloca
define void @scalable_alloca() #1 {
  %local0 = alloca <vscale x 16 x i8>
  load volatile <vscale x 16 x i8>, <vscale x 16 x i8>* %local0
  ret void
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction{{.*}}asm_indirect_output
; FALLBACK-WITH-REPORT-OUT-LABEL: asm_indirect_output
define void @asm_indirect_output() {
entry:
  %ap = alloca i8*, align 8
  %0 = load i8*, i8** %ap, align 8
  call void asm sideeffect "", "=*r|m,0,~{memory}"(i8** %ap, i8* %0)
  ret void
}

attributes #1 = { "target-features"="+sve" }

declare <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 %pattern)
declare <vscale x 16 x i8> @llvm.aarch64.sve.ld1.nxv16i8(<vscale x 16 x i1>, i8*)
