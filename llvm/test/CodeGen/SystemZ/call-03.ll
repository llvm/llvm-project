; Test sibling calls.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @ok(i8 %r2, i16 %r3, i32 %r4, i64 %r5, float %f0, double %f2,
                 float %f4, double %f6)
declare void @uses_r6(i8 %r2, i16 %r3, i32 %r4, i64 %r5, i64 %r6)
declare void @uses_indirect(fp128 %r2)
declare void @uses_stack(float %f0, float %f2, float %f4, float %f6,
                         float %stack)
declare i32 @returns_i32()
declare i64 @returns_i64()

; Check the maximum number of arguments that we can pass and still use
; a sibling call.
define void @f1() {
; CHECK-LABEL: f1:
; CHECK-DAG: lzer %f0
; CHECK-DAG: lzdr %f2
; CHECK-DAG: lhi %r2, 1
; CHECK-DAG: lhi %r3, 2
; CHECK-DAG: lhi %r4, 3
; CHECK-DAG: lghi %r5, 4
; CHECK-DAG: {{ler %f4, %f0|lzer %f4}}
; CHECK-DAG: {{ldr %f6, %f2|lzdr %f6}}
; CHECK: jg ok@PLT
  tail call void @ok(i8 1, i16 2, i32 3, i64 4, float 0.0, double 0.0,
                     float 0.0, double 0.0)
  ret void
}

; Check a call that uses %r6 to pass an argument.  At the moment we don't
; use sibling calls in that case.
define void @f2() {
; CHECK-LABEL: f2:
; CHECK: brasl %r14, uses_r6@PLT
; CHECK: br %r14
  tail call void @uses_r6(i8 1, i16 2, i32 3, i64 4, i64 5)
  ret void
}

; Check a call that passes indirect arguments.  We can't use sibling
; calls in that case.
define void @f3() {
; CHECK-LABEL: f3:
; CHECK: brasl %r14, uses_indirect@PLT
; CHECK: br %r14
  tail call void @uses_indirect(fp128 0xL00000000000000000000000000000000)
  ret void
}

; Check a call that uses direct stack arguments, which again prevents
; sibling calls
define void @f4() {
; CHECK-LABEL: f4:
; CHECK: brasl %r14, uses_stack@PLT
; CHECK: br %r14
  tail call void @uses_stack(float 0.0, float 0.0, float 0.0, float 0.0,
                             float 0.0)
  ret void
}

; Check an indirect call.  In this case the only acceptable choice for
; the target register is %r1.
define void @f5(ptr %foo) {
; CHECK-LABEL: f5:
; CHECK: lgr %r1, %r2
; CHECK-DAG: lhi %r2, 1
; CHECK-DAG: lhi %r3, 2
; CHECK-DAG: lhi %r4, 3
; CHECK-DAG: lhi %r5, 4
; CHECK: br %r1
  tail call void %foo(i32 1, i32 2, i32 3, i32 4)
  ret void
}

; Check an indirect call that will be forced into a call-saved GPR
; (which should be %r13, the highest GPR not used for anything else).
define void @f6(ptr %foo) {
; CHECK-LABEL: f6:
; CHECK: stmg %r13, %r15, 104(%r15)
; CHECK: lgr %r13, %r2
; CHECK: brasl %r14, returns_i32
; CHECK: lgr %r1, %r13
; CHECK: lmg %r13, %r15, 264(%r15)
; CHECK: br %r1
  %arg = call i32 @returns_i32()
  tail call void %foo(i32 %arg)
  ret void
}

; Test a function that returns a value.
define i64 @f7() {
; CHECK-LABEL: f7:
; CHECK: jg returns_i64@PLT
  %res = tail call i64 @returns_i64()
  ret i64 %res
}

; Test a function that returns a value truncated from i64 to i32.
define i32 @f8() {
; CHECK-LABEL: f8:
; CHECK: jg returns_i64@PLT
  %res = tail call i64 @returns_i64()
  %trunc = trunc i64 %res to i32
  ret i32 %trunc
}

; Test a function that returns a value truncated from i64 to i7.
define i7 @f9() {
; CHECK-LABEL: f9:
; CHECK: jg returns_i64@PLT
  %res = tail call i64 @returns_i64()
  %trunc = trunc i64 %res to i7
  ret i7 %trunc
}

; Test a function that returns a value truncated from i32 to i8.
define i8 @f10() {
; CHECK-LABEL: f10:
; CHECK: jg returns_i32@PLT
  %res = tail call i32 @returns_i32()
  %trunc = trunc i32 %res to i8
  ret i8 %trunc
}
