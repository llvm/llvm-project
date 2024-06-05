; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-link %t/1.ll %t/2.ll -S -o - | FileCheck %s
; CHECK: {i32 2, !"darwin.target_variant.triple", !"x86_64-apple-ios13.1-macabi"}

; RUN: llvm-link %t/1.ll %t/old.ll -S -o - | FileCheck %s -check-prefix OLD
; OLD: {i32 4, !"darwin.target_variant.triple", !"x86_64-apple-ios14.0-macabi"}

;--- 1.ll
target triple = "x86_64-apple-macos10.15";
!llvm.module.flags = !{!0, !1, !2};
!0 = !{i32 2, !"SDK Version", [3 x i32] [ i32 10, i32 15, i32 1 ] };
!1 = !{i32 2, !"darwin.target_variant.triple", !"x86_64-apple-ios13.1-macabi"};
!2 = !{i32 2, !"darwin.target_variant.SDK Version", [2 x i32] [ i32 13, i32 2 ] };

define void @foo() {
entry:
  ret void
}

;--- 2.ll
target triple = "x86_64-apple-macos10.15";
!llvm.module.flags = !{!0, !1, !2};
!0 = !{i32 2, !"SDK Version", [3 x i32] [ i32 10, i32 15, i32 1 ] };
!1 = !{i32 2, !"darwin.target_variant.triple", !"x86_64-apple-ios14.0-macabi"};
!2 = !{i32 2, !"darwin.target_variant.SDK Version", [2 x i32] [ i32 13, i32 2 ] };

define void @bar() {
entry:
  ret void
}

;--- old.ll
target triple = "x86_64-apple-macos10.15";
!llvm.module.flags = !{!0, !1, !2};
!0 = !{i32 2, !"SDK Version", [3 x i32] [ i32 10, i32 15, i32 1 ] };
!1 = !{i32 4, !"darwin.target_variant.triple", !"x86_64-apple-ios14.0-macabi"};
!2 = !{i32 2, !"darwin.target_variant.SDK Version", [2 x i32] [ i32 13, i32 2 ] };

define void @old() {
entry:
  ret void
}
