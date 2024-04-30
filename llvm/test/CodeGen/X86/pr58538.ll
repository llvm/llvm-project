; RUN: opt -passes='require<profile-summary>,function(codegenprepare)' -mtriple=x86_64 %s -S -o - | FileCheck %s
; RUN: opt -passes='require<profile-summary>,function(codegenprepare)' -mtriple=i386 %s -S -o - | FileCheck %s

define i32 @f(i32 %0) {
; CHECK-LABEL: @f
; CHECK: BB:
; CHECK:   %P0 = alloca i32, i32 8, align 4
; CHECK:   %P1 = getelementptr i32, ptr %P0, i32 1
; CHECK:   %1 = icmp eq i32 %0, 0
; CHECK:   %P2 = getelementptr i1, ptr %P1, i1 %1
; CHECK:   %2 = icmp eq i32 %0, 0
; CHECK:   %P3 = select i1 %2, ptr %P1, ptr %P2
; CHECK:   %L1 = load i32, ptr %P3, align 4
; CHECK:   ret i32 %L1
BB:
  %P0 = alloca i32, i32 8
  %P1 = getelementptr i32, ptr %P0, i32 1
  %B0 = icmp eq i32 %0, 0
  br label %BB1

BB1:                                              ; preds = %BB1, %BB
  %P2 = getelementptr i1, ptr %P1, i1 %B0
  br i1 false, label %BB1, label %BB2

BB2:                                              ; preds = %BB1
  %P3 = select i1 %B0, ptr %P1, ptr %P2
  %L1 = load i32, ptr %P3
  ret i32 %L1
}
