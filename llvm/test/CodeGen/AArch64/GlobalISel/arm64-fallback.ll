; RUN: not llc -O0 -global-isel -global-isel-abort=1 -verify-machineinstrs %s -o - 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: llc -O0 -global-isel -global-isel-abort=0 -verify-machineinstrs %s -o - 2>&1 | FileCheck %s --check-prefix=FALLBACK
; RUN: llc -O0 -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' -verify-machineinstrs %s -o %t.out 2> %t.err
; RUN: FileCheck %s --check-prefix=FALLBACK-WITH-REPORT-OUT < %t.out
; RUN: FileCheck %s --check-prefix=FALLBACK-WITH-REPORT-ERR < %t.err
; RUN: not llc -global-isel -mtriple aarch64_be %s -o - 2>&1 | FileCheck %s --check-prefix=BIG-ENDIAN
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
; ERROR: unable to lower arguments: i128 (i128)* (in function: ABIi128)
; FALLBACK: ldr q0,
; FALLBACK-NEXT: bl __fixunstfti
;
; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to lower arguments: i128 (i128)* (in function: ABIi128)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for ABIi128
; FALLBACK-WITH-REPORT-OUT-LABEL: ABIi128:
; FALLBACK-WITH-REPORT-OUT: ldr q0,
; FALLBACK-WITH-REPORT-OUT-NEXT: bl __fixunstfti
define i128 @ABIi128(i128 %arg1) {
  %farg1 =       bitcast i128 %arg1 to fp128
  %res = fptoui fp128 %farg1 to i128
  ret i128 %res
}

; It happens that we don't handle ConstantArray instances yet during
; translation. Any other constant would be fine too.

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate constant: [1 x double] (in function: constant)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for constant
; FALLBACK-WITH-REPORT-OUT-LABEL: constant:
; FALLBACK-WITH-REPORT-OUT: fmov d0, #1.0
define [1 x double] @constant() {
  ret [1 x double] [double 1.0]
}

  ; The key problem here is that we may fail to create an MBB referenced by a
  ; PHI. If so, we cannot complete the G_PHI and mustn't try or bad things
  ; happen.
; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: cannot select: G_STORE %6(s32), %2(p0); mem:ST4[%addr] GPR:%6,%2 (in function: pending_phis)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for pending_phis
; FALLBACK-WITH-REPORT-OUT-LABEL: pending_phis:
define i32 @pending_phis(i1 %tst, i32 %val, i32* %addr) {
  br i1 %tst, label %true, label %false

end:
  %res = phi i32 [%val, %true], [42, %false]
  ret i32 %res

true:
  store atomic i32 42, i32* %addr seq_cst, align 4
  br label %end

false:
  br label %end

}

  ; General legalizer inability to handle types whose size wasn't a power of 2.
; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: G_STORE %1(s42), %0(p0); mem:ST6[%addr](align=8) (in function: odd_type)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for odd_type
; FALLBACK-WITH-REPORT-OUT-LABEL: odd_type:
define void @odd_type(i42* %addr) {
  %val42 = load i42, i42* %addr
  store i42 %val42, i42* %addr
  ret void
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: G_STORE %1(<7 x s32>), %0(p0); mem:ST28[%addr](align=32) (in function: odd_vector)
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

; Just to make sure we don't accidentally emit a normal load/store.
; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: cannot select: %2:gpr(s64) = G_LOAD %0(p0); mem:LD8[%addr] GPR:%2,%0 (in function: atomic_ops)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for atomic_ops
; FALLBACK-WITH-REPORT-LABEL: atomic_ops:
define i64 @atomic_ops(i64* %addr) {
  store atomic i64 0, i64* %addr unordered, align 8
  %res = load atomic i64, i64* %addr seq_cst, align 8
  ret i64 %res
}

; Make sure we don't mess up metadata arguments.
declare void @llvm.write_register.i64(metadata, i64)

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction: call: ' call void @llvm.write_register.i64(metadata !0, i64 0)' (in function: test_write_register_intrin)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for test_write_register_intrin
; FALLBACK-WITH-REPORT-LABEL: test_write_register_intrin:
define void @test_write_register_intrin() {
  call void @llvm.write_register.i64(metadata !{!"sp"}, i64 0)
  ret void
}

@_ZTIi = external global i8*
declare i32 @__gxx_personality_v0(...)

; Check that we fallback on invoke translation failures.
; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to translate instruction: invoke: '  invoke void %callee(i128 0)
; FALLBACK-WITH-REPORT-NEXT:   to label %continue unwind label %broken' (in function: invoke_weird_type)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for invoke_weird_type
; FALLBACK-WITH-REPORT-OUT-LABEL: invoke_weird_type:
define void @invoke_weird_type(void(i128)* %callee) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  invoke void %callee(i128 0)
    to label %continue unwind label %broken

broken:
  landingpad { i8*, i32 } catch i8* bitcast(i8** @_ZTIi to i8*)
  ret void

continue:
  ret void
}

; Check that we fallback on invoke translation failures.
; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: %0:_(s128) = G_FCONSTANT fp128 0xL00000000000000004000000000000000
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for test_quad_dump
; FALLBACK-WITH-REPORT-OUT-LABEL: test_quad_dump:
define fp128 @test_quad_dump() {
  ret fp128 0xL00000000000000004000000000000000
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: %0:_(p0) = G_EXTRACT_VECTOR_ELT %1(<2 x p0>), %2(s32); (in function: vector_of_pointers_extractelement)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for vector_of_pointers_extractelement
; FALLBACK-WITH-REPORT-OUT-LABEL: vector_of_pointers_extractelement:
@var = global <2 x i16*> zeroinitializer
define void @vector_of_pointers_extractelement() {
  br label %end

block:
  %dummy = extractelement <2 x i16*> %vec, i32 0
  store i16* %dummy, i16** undef
  ret void

end:
  %vec = load <2 x i16*>, <2 x i16*>* undef
  br label %block
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: G_STORE %0(<2 x p0>), %4(p0); mem:ST16[undef] (in function: vector_of_pointers_insertelement)
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

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: G_STORE %1(s96), %3(p0); mem:ST12[undef](align=4) (in function: nonpow2_insertvalue_narrowing)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for nonpow2_insertvalue_narrowing
; FALLBACK-WITH-REPORT-OUT-LABEL: nonpow2_insertvalue_narrowing:
%struct96 = type { float, float, float }
define void @nonpow2_insertvalue_narrowing(float %a) {
  %dummy = insertvalue %struct96 undef, float %a, 0
  store %struct96 %dummy, %struct96* undef
  ret void
}

; FALLBACK-WITH-REPORT-ERR remark: <unknown>:0:0: unable to legalize instruction: G_STORE %3, %4; mem:ST12[undef](align=16) (in function: nonpow2_add_narrowing)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for nonpow2_add_narrowing
; FALLBACK-WITH-REPORT-OUT-LABEL: nonpow2_add_narrowing:
define void @nonpow2_add_narrowing() {
  %a = add i128 undef, undef
  %b = trunc i128 %a to i96
  %dummy = add i96 %b, %b
  store i96 %dummy, i96* undef
  ret void
}

; FALLBACK-WITH-REPORT-ERR remark: <unknown>:0:0: unable to legalize instruction: G_STORE %3, %4; mem:ST12[undef](align=16) (in function: nonpow2_add_narrowing)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for nonpow2_or_narrowing
; FALLBACK-WITH-REPORT-OUT-LABEL: nonpow2_or_narrowing:
define void @nonpow2_or_narrowing() {
  %a = add i128 undef, undef
  %b = trunc i128 %a to i96
  %dummy = or i96 %b, %b
  store i96 %dummy, i96* undef
  ret void
}

; FALLBACK-WITH-REPORT-ERR remark: <unknown>:0:0: unable to legalize instruction: G_STORE %0, %1; mem:ST12[undef](align=16) (in function: nonpow2_load_narrowing)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for nonpow2_load_narrowing
; FALLBACK-WITH-REPORT-OUT-LABEL: nonpow2_load_narrowing:
define void @nonpow2_load_narrowing() {
  %dummy = load i96, i96* undef
  store i96 %dummy, i96* undef
  ret void
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: G_STORE %3(s96), %0(p0); mem:ST12[%c](align=16) (in function: nonpow2_store_narrowing
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for nonpow2_store_narrowing
; FALLBACK-WITH-REPORT-OUT-LABEL: nonpow2_store_narrowing:
define void @nonpow2_store_narrowing(i96* %c) {
  %a = add i128 undef, undef
  %b = trunc i128 %a to i96
  store i96 %b, i96* %c
  ret void
}

; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: G_STORE %0(s96), %1(p0); mem:ST12[undef](align=16) (in function: nonpow2_constant_narrowing)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for nonpow2_constant_narrowing
; FALLBACK-WITH-REPORT-OUT-LABEL: nonpow2_constant_narrowing:
define void @nonpow2_constant_narrowing() {
  store i96 0, i96* undef
  ret void
}

; Currently can't handle vector lengths that aren't an exact multiple of
; natively supported vector lengths. Test that the fall-back works for those.
; FALLBACK-WITH-REPORT-ERR-G_IMPLICIT_DEF-LEGALIZABLE: (FIXME: this is what is expected once we can legalize non-pow-of-2 G_IMPLICIT_DEF) remark: <unknown>:0:0: unable to legalize instruction: %1(<7 x s64>) = G_ADD %0, %0; (in function: nonpow2_vector_add_fewerelements
; FALLBACK-WITH-REPORT-ERR: remark: <unknown>:0:0: unable to legalize instruction: %2:_(s64) = G_EXTRACT_VECTOR_ELT %1(<7 x s64>), %3(s64); (in function: nonpow2_vector_add_fewerelements)
; FALLBACK-WITH-REPORT-ERR: warning: Instruction selection used fallback path for nonpow2_vector_add_fewerelements
; FALLBACK-WITH-REPORT-OUT-LABEL: nonpow2_vector_add_fewerelements:
define void @nonpow2_vector_add_fewerelements() {
  %dummy = add <7 x i64> undef, undef
  %ex = extractelement <7 x i64> %dummy, i64 0
  store i64 %ex, i64* undef
  ret void
}
