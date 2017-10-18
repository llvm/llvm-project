; RUN: llc -mtriple=arm64-apple-ios < %s | FileCheck %s
; RUN: llc -mtriple=arm64-linux-gnu < %s | FileCheck %s --check-prefix=CHECK-LINUX
; <rdar://11417675>

define void @sum(i32 %a, i32* %to, i32 %c) {
entry:
  switch i32 %a, label %exit [
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
  ]
bb1:
  %b = add i32 %c, 1
  store i32 %b, i32* %to
  br label %exit
bb2:
  store i32 2, i32* %to
  br label %exit
bb3:
  store i32 3, i32* %to
  br label %exit
bb4:
  store i32 5, i32* %to
  br label %exit
exit:
  ret void
}

; CHECK-LABEL: sum:
; CHECK: adrp    {{x[0-9]+}}, LJTI0_0@PAGE
; CHECK:  add    {{x[0-9]+}}, {{x[0-9]+}}, LJTI0_0@PAGEOFF

; CHECK-LINUX-LABEL: sum:
; CHECK-LINUX: adrp    {{x[0-9]+}}, .LJTI0_0
; CHECK-LINUX:  add    {{x[0-9]+}}, {{x[0-9]+}}, :lo12:.LJTI0_0
