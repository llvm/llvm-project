; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; Test atomic reduction instruction parsing and printing

; Integer atomic reductions - all operations
; CHECK-LABEL: @test_atomic_reduction_ops
define void @test_atomic_reduction_ops(ptr %ptr, i32 %val, i64 %val64, float %fval, double %dval) {
  ; CHECK: storermw add ptr %ptr, i32 %val seq_cst, align 4
  storermw add ptr %ptr, i32 %val seq_cst, align 4

  ; CHECK: storermw sub ptr %ptr, i32 %val seq_cst, align 4
  storermw sub ptr %ptr, i32 %val seq_cst, align 4

  ; CHECK: storermw and ptr %ptr, i32 %val seq_cst, align 4
  storermw and ptr %ptr, i32 %val seq_cst, align 4

  ; CHECK: storermw nand ptr %ptr, i32 %val seq_cst, align 4
  storermw nand ptr %ptr, i32 %val seq_cst, align 4

  ; CHECK: storermw or ptr %ptr, i32 %val seq_cst, align 4
  storermw or ptr %ptr, i32 %val seq_cst, align 4

  ; CHECK: storermw xor ptr %ptr, i32 %val seq_cst, align 4
  storermw xor ptr %ptr, i32 %val seq_cst, align 4

  ; CHECK: storermw max ptr %ptr, i32 %val seq_cst, align 4
  storermw max ptr %ptr, i32 %val seq_cst, align 4

  ; CHECK: storermw min ptr %ptr, i32 %val seq_cst, align 4
  storermw min ptr %ptr, i32 %val seq_cst, align 4

  ; CHECK: storermw umax ptr %ptr, i32 %val seq_cst, align 4
  storermw umax ptr %ptr, i32 %val seq_cst, align 4

  ; CHECK: storermw umin ptr %ptr, i32 %val seq_cst, align 4
  storermw umin ptr %ptr, i32 %val seq_cst, align 4

  ; CHECK: storermw uinc_wrap ptr %ptr, i32 %val seq_cst, align 4
  storermw uinc_wrap ptr %ptr, i32 %val seq_cst, align 4

  ; CHECK: storermw udec_wrap ptr %ptr, i32 %val seq_cst, align 4
  storermw udec_wrap ptr %ptr, i32 %val seq_cst, align 4

  ; CHECK: storermw usub_cond ptr %ptr, i32 %val seq_cst, align 4
  storermw usub_cond ptr %ptr, i32 %val seq_cst, align 4

  ; CHECK: storermw usub_sat ptr %ptr, i32 %val seq_cst, align 4
  storermw usub_sat ptr %ptr, i32 %val seq_cst, align 4

  ; 64-bit integer
  ; CHECK: storermw add ptr %ptr, i64 %val64 seq_cst, align 8
  storermw add ptr %ptr, i64 %val64 seq_cst, align 8

  ; Floating-point atomic reductions
  ; CHECK: storermw fadd ptr %ptr, float %fval seq_cst, align 4
  storermw fadd ptr %ptr, float %fval seq_cst, align 4

  ; CHECK: storermw fsub ptr %ptr, float %fval seq_cst, align 4
  storermw fsub ptr %ptr, float %fval seq_cst, align 4

  ; CHECK: storermw fmax ptr %ptr, float %fval seq_cst, align 4
  storermw fmax ptr %ptr, float %fval seq_cst, align 4

  ; CHECK: storermw fmin ptr %ptr, float %fval seq_cst, align 4
  storermw fmin ptr %ptr, float %fval seq_cst, align 4

  ; CHECK: storermw fmaximum ptr %ptr, float %fval seq_cst, align 4
  storermw fmaximum ptr %ptr, float %fval seq_cst, align 4

  ; CHECK: storermw fminimum ptr %ptr, float %fval seq_cst, align 4
  storermw fminimum ptr %ptr, float %fval seq_cst, align 4

  ; CHECK: storermw fmaximumnum ptr %ptr, float %fval seq_cst, align 4
  storermw fmaximumnum ptr %ptr, float %fval seq_cst, align 4

  ; CHECK: storermw fminimumnum ptr %ptr, float %fval seq_cst, align 4
  storermw fminimumnum ptr %ptr, float %fval seq_cst, align 4

  ; CHECK: storermw fmaximum ptr %ptr, double %dval seq_cst, align 8
  storermw fmaximum ptr %ptr, double %dval seq_cst, align 8

  ; CHECK: storermw fminimum ptr %ptr, double %dval seq_cst, align 8
  storermw fminimum ptr %ptr, double %dval seq_cst, align 8

  ret void
}

; Test different integer sizes (i8, i16)
; CHECK-LABEL: @test_sizes
define void @test_sizes(ptr %ptr, i8 %v8, i16 %v16) {
  ; CHECK: storermw add ptr %ptr, i8 %v8 seq_cst, align 1
  storermw add ptr %ptr, i8 %v8 seq_cst, align 1

  ; CHECK: storermw add ptr %ptr, i16 %v16 seq_cst, align 2
  storermw add ptr %ptr, i16 %v16 seq_cst, align 2

  ret void
}

; Test different orderings (monotonic, release, seq_cst)
; CHECK-LABEL: @test_orderings
define void @test_orderings(ptr %ptr, i32 %val) {
  ; CHECK: storermw add ptr %ptr, i32 %val monotonic, align 4
  storermw add ptr %ptr, i32 %val monotonic, align 4

  ; CHECK: storermw add ptr %ptr, i32 %val release, align 4
  storermw add ptr %ptr, i32 %val release, align 4

  ; CHECK: storermw add ptr %ptr, i32 %val seq_cst, align 4
  storermw add ptr %ptr, i32 %val seq_cst, align 4

  ret void
}

; Test volatile flag
; CHECK-LABEL: @test_volatile
define void @test_volatile(ptr %ptr, i32 %val) {
  ; CHECK: storermw volatile add ptr %ptr, i32 %val seq_cst, align 4
  storermw volatile add ptr %ptr, i32 %val seq_cst, align 4

  ret void
}

; Test sync scope
; CHECK-LABEL: @test_syncscope
define void @test_syncscope(ptr %ptr, i32 %val) {
  ; CHECK: storermw add ptr %ptr, i32 %val syncscope("singlethread") seq_cst, align 4
  storermw add ptr %ptr, i32 %val syncscope("singlethread") seq_cst, align 4

  ; CHECK: storermw add ptr %ptr, i32 %val syncscope("agent") monotonic, align 4
  storermw add ptr %ptr, i32 %val syncscope("agent") monotonic, align 4

  ret void
}

; Test default alignment: when the ", align N" clause is omitted, the parser
; sets align to the size of the value type; the printer then prints it back.
; CHECK-LABEL: @test_default_alignment
define void @test_default_alignment(ptr %ptr, i8 %v8, i16 %v16, i32 %v32,
                                    i64 %v64, float %vf, double %vd,
                                    <4 x i32> %v4i32) {
  ; CHECK: storermw add ptr %ptr, i8 %v8 monotonic, align 1
  storermw add ptr %ptr, i8 %v8 monotonic

  ; CHECK: storermw add ptr %ptr, i16 %v16 monotonic, align 2
  storermw add ptr %ptr, i16 %v16 monotonic

  ; CHECK: storermw add ptr %ptr, i32 %v32 monotonic, align 4
  storermw add ptr %ptr, i32 %v32 monotonic

  ; CHECK: storermw add ptr %ptr, i64 %v64 monotonic, align 8
  storermw add ptr %ptr, i64 %v64 monotonic

  ; CHECK: storermw fadd ptr %ptr, float %vf monotonic, align 4
  storermw fadd ptr %ptr, float %vf monotonic

  ; CHECK: storermw fadd ptr %ptr, double %vd monotonic, align 8
  storermw fadd ptr %ptr, double %vd monotonic

  ; CHECK: storermw elementwise add ptr %ptr, <4 x i32> %v4i32 monotonic, align 16
  storermw elementwise add ptr %ptr, <4 x i32> %v4i32 monotonic

  ret void
}

; Test volatile + syncscope combination
; CHECK-LABEL: @test_volatile_syncscope
define void @test_volatile_syncscope(ptr %ptr, i32 %val) {
  ; CHECK: storermw volatile add ptr %ptr, i32 %val syncscope("singlethread") release, align 4
  storermw volatile add ptr %ptr, i32 %val syncscope("singlethread") release, align 4

  ret void
}

; Test elementwise modifier round-trip
; CHECK-LABEL: @test_elementwise
define void @test_elementwise(ptr %ptr, <4 x i32> %v4i32, <2 x i64> %v2i64,
                              <4 x float> %v4f32) {
  ; CHECK: storermw elementwise add ptr %ptr, <4 x i32> %v4i32 seq_cst, align 16
  storermw elementwise add ptr %ptr, <4 x i32> %v4i32 seq_cst, align 16

  ; CHECK: storermw elementwise umax ptr %ptr, <2 x i64> %v2i64 monotonic, align 16
  storermw elementwise umax ptr %ptr, <2 x i64> %v2i64 monotonic, align 16

  ; CHECK: storermw elementwise fadd ptr %ptr, <4 x float> %v4f32 release, align 16
  storermw elementwise fadd ptr %ptr, <4 x float> %v4f32 release, align 16

  ; CHECK: storermw volatile elementwise add ptr %ptr, <4 x i32> %v4i32 seq_cst, align 16
  storermw volatile elementwise add ptr %ptr, <4 x i32> %v4i32 seq_cst, align 16

  ret void
}
