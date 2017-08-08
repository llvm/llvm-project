; RUN: llc -verify-machineinstrs -print-before=peephole-opt -print-after=peephole-opt -mtriple=powerpc64-unknown-linux-gnu -o /dev/null 2>&1 < %s | FileCheck %s

; CHECK-LABEL: fn1
define signext i32 @fn1(i32 %baz) {
  %1 = mul nsw i32 %baz, 208
  %2 = zext i32 %1 to i64
  %3 = shl i64 %2, 48
  %4 = ashr exact i64 %3, 48
; CHECK: ANDIo8 {{[^,]+}}, 65520, %CR0<imp-def,dead>;
; CHECK: CMPLDI
; CHECK: BCC

; CHECK: ANDIo8 {{[^,]+}}, 65520, %CR0<imp-def>;
; CHECK: COPY %CR0
; CHECK: BCC
  %5 = icmp eq i64 %4, 0
  br i1 %5, label %foo, label %bar

foo:
  ret i32 1

bar:
  ret i32 0
}

; CHECK-LABEL: fn2
define signext i32 @fn2(i64 %a, i64 %b) {
; CHECK: OR8o {{[^, ]+}}, {{[^, ]+}}, %CR0<imp-def>;
; CHECK: [[CREG:[^, ]+]]<def> = COPY %CR0
; CHECK: BCC 12, [[CREG]]<kill>
  %1 = or i64 %b, %a
  %2 = icmp sgt i64 %1, -1
  br i1 %2, label %foo, label %bar

foo:
  ret i32 1

bar:
  ret i32 0
}

; CHECK-LABEL: fn3
define signext i32 @fn3(i32 %a) {
; CHECK: ANDIo {{[^, ]+}}, 10, %CR0<imp-def>;
; CHECK: [[CREG:[^, ]+]]<def> = COPY %CR0
; CHECK: BCC 76, [[CREG]]<kill>
  %1 = and i32 %a, 10
  %2 = icmp ne i32 %1, 0
  br i1 %2, label %foo, label %bar

foo:
  ret i32 1

bar:
  ret i32 0
}
