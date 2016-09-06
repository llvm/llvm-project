; RUN: llc -mtriple=armv7-unknown-linux < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-ARM %s
; RUN: llc -mtriple=thumbv6-unknown-linux < %s | FileCheck --check-prefix=CHECK-THUMB1 %s
; RUN: llc -mtriple=thumbv7-unknown-linux < %s | FileCheck --check-prefix=CHECK --check-prefix=CHECK-THUMB2 %s

; CHECK-THUMB1-NOT: sbc

; CHECK-LABEL: test_slt1:
define i32 @test_slt1(i64 %a, i64 %b) {
entry:
  ; CHECK-ARM: subs {{[^,]+}}, r0, r2
  ; CHECK-ARM: mov [[TMP:[0-9a-z]+]], #2
  ; CHECK-ARM: sbcs {{[^,]+}}, r1, r3
  ; CHECK-ARM: movwlt [[TMP]], #1
  ; CHECK-ARM: mov r0, [[TMP]]
  ; CHECK-ARM: bx lr
  ; CHECK-THUMB2: subs {{[^,]+}}, r0, r2
  ; CHECK-THUMB2: mov.w [[TMP:[0-9a-z]+]], #2
  ; CHECK-THUMB2: sbcs.w {{[^,]+}}, r1, r3
  ; CHECK-THUMB2: it lt
  ; CHECK-THUMB2: movlt.w [[TMP]], #1
  ; CHECK-THUMB2: mov r0, [[TMP]]
  ; CHECK-THUMB2: bx lr
  %cmp = icmp slt i64 %a, %b
  br i1 %cmp, label %bb1, label %bb2
bb1:
  ret i32 1
bb2:
  ret i32 2
}

; CHECK-LABEL: test_slt2:
define void @test_slt2(i64 %a, i64 %b) {
entry:
  %cmp = icmp slt i64 %a, %b
  ; CHECK-ARM: subs {{[^,]+}}, r0, r2
  ; CHECK-ARM: sbcs {{[^,]+}}, r1, r3
  ; CHECK-THUMB2: subs {{[^,]+}}, r0, r2
  ; CHECK-THUMB2: sbcs.w {{[^,]+}}, r1, r3
  ; CHECK: bge [[BB2:\.[0-9A-Za-z_]+]]
  br i1 %cmp, label %bb1, label %bb2
bb1:
  call void @f()
  ret void
bb2:
  ; CHECK: [[BB2]]:
  ; CHECK-NEXT: bl g
  call void @g()
  ret void
}

declare void @f()
declare void @g()
