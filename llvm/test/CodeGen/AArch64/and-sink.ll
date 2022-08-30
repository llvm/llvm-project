; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs < %s | FileCheck %s
; RUN: opt -S -codegenprepare -mtriple=aarch64-linux %s | FileCheck --check-prefix=CHECK-CGP %s
; RUN: opt -S -codegenprepare -cgpp-huge-func=0 -mtriple=aarch64-linux %s | FileCheck --check-prefix=CHECK-CGP %s

@A = dso_local global i32 zeroinitializer
@B = dso_local global i32 zeroinitializer
@C = dso_local global i32 zeroinitializer

; Test that and is sunk into cmp block to form tbz.
define dso_local i32 @and_sink1(i32 %a, i1 %c) {
; CHECK-LABEL: and_sink1:
; CHECK: tbz w1, #0
; CHECK: str wzr, [x{{[0-9]+}}, :lo12:A]
; CHECK: tbnz {{w[0-9]+}}, #2

; CHECK-CGP-LABEL: @and_sink1(
; CHECK-CGP-NOT: and i32
  %and = and i32 %a, 4
  br i1 %c, label %bb0, label %bb2
bb0:
; CHECK-CGP-LABEL: bb0:
; CHECK-CGP: and i32
; CHECK-CGP-NEXT: icmp eq i32
; CHECK-CGP-NEXT: store
; CHECK-CGP-NEXT: br
  %cmp = icmp eq i32 %and, 0
  store i32 0, i32* @A
  br i1 %cmp, label %bb1, label %bb2
bb1:
  ret i32 1
bb2:
  ret i32 0
}

; Test that both 'and' and cmp get sunk to form tbz.
define dso_local i32 @and_sink2(i32 %a, i1 %c, i1 %c2) {
; CHECK-LABEL: and_sink2:
; CHECK: str wzr, [x{{[0-9]+}}, :lo12:A]
; CHECK: tbz w1, #0
; CHECK: str wzr, [x{{[0-9]+}}, :lo12:B]
; CHECK: tbz w2, #0
; CHECK: str wzr, [x{{[0-9]+}}, :lo12:C]
; CHECK: tbnz {{w[0-9]+}}, #2

; CHECK-CGP-LABEL: @and_sink2(
; CHECK-CGP-NOT: and i32
  %and = and i32 %a, 4
  store i32 0, i32* @A
  br i1 %c, label %bb0, label %bb3
bb0:
; CHECK-CGP-LABEL: bb0:
; CHECK-CGP-NOT: and i32
; CHECK-CGP-NOT: icmp
  %cmp = icmp eq i32 %and, 0
  store i32 0, i32* @B
  br i1 %c2, label %bb1, label %bb3
bb1:
; CHECK-CGP-LABEL: bb1:
; CHECK-CGP: and i32
; CHECK-CGP-NEXT: icmp eq i32
; CHECK-CGP-NEXT: store
; CHECK-CGP-NEXT: br
  store i32 0, i32* @C
  br i1 %cmp, label %bb2, label %bb0
bb2:
  ret i32 1
bb3:
  ret i32 0
}

; Test that 'and' is not sunk since cbz is a better alternative.
define dso_local i32 @and_sink3(i32 %a) {
; CHECK-LABEL: and_sink3:
; CHECK: and [[REG:w[0-9]+]], w0, #0x3
; CHECK: [[LOOP:.L[A-Z0-9_]+]]:
; CHECK: str wzr, [x{{[0-9]+}}, :lo12:A]
; CHECK: cbz [[REG]], [[LOOP]]

; CHECK-CGP-LABEL: @and_sink3(
; CHECK-CGP-NEXT: and i32
  %and = and i32 %a, 3
  br label %bb0
bb0:
; CHECK-CGP-LABEL: bb0:
; CHECK-CGP-NOT: and i32
  %cmp = icmp eq i32 %and, 0
  store i32 0, i32* @A
  br i1 %cmp, label %bb0, label %bb2
bb2:
  ret i32 0
}
