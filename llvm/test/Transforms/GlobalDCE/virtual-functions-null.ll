; RUN: opt < %s -globaldce -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare { i8*, i1 } @llvm.type.checked.load(i8*, i32, metadata)

@vtableA = internal unnamed_addr constant { [2 x i8*] } { [2 x i8*] [
  i8* null,
  i8* bitcast (void ()* @vfunc2 to i8*)
]}, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 8, !"vfunc2.type"}, !vcall_visibility !{i64 2}

; CHECK:      @vtableA = internal unnamed_addr constant { [2 x i8*] } { [2 x i8*] [
; CHECK-SAME:   i8* null,
; CHECK-SAME:   i8* bitcast (void ()* @vfunc2 to i8*)
; CHECK-SAME: ] }, align 8

@vtableB = internal unnamed_addr constant { [2 x i8*] } { [2 x i8*] [
  i8* bitcast (void ()* @vfunc1 to i8*),
  i8* bitcast (void ()* @vfunc2 to i8*)
]}, align 8, !type !{i64 0, !"vfunc1.type"}, !type !{i64 8, !"vfunc2.type"}, !vcall_visibility !{i64 2}

; CHECK:      @vtableB = internal unnamed_addr constant { [2 x i8*] } { [2 x i8*] [
; CHECK-SAME:   i8* null,
; CHECK-SAME:   i8* bitcast (void ()* @vfunc2 to i8*)
; CHECK-SAME: ] }, align 8

define internal void @vfunc1() {
  ret void
}

define internal void @vfunc2() {
  ret void
}

define void @main() {
  %1 = ptrtoint { [2 x i8*] }* @vtableA to i64 ; to keep @vtableA alive
  %2 = ptrtoint { [2 x i8*] }* @vtableB to i64 ; to keep @vtableB alive
  %3 = tail call { i8*, i1 } @llvm.type.checked.load(i8* null, i32 0, metadata !"vfunc1.type")
  %4 = tail call { i8*, i1 } @llvm.type.checked.load(i8* null, i32 0, metadata !"vfunc2.type")
  ret void
}

!999 = !{i32 1, !"Virtual Function Elim", i32 1}
!llvm.module.flags = !{!999}
