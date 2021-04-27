; RUN: llvm-link %s -S -o - | FileCheck %s
; RUN: llvm-link %s %S/Inputs/metadata-attach.ll -S -o - | FileCheck --check-prefix=CHECK-LINKED1 %s
; RUN: llvm-link %S/Inputs/metadata-attach.ll %s -S -o - | FileCheck --check-prefix=CHECK-LINKED2 %s

; CHECK: @g1 = global i32 0, !attach !0{{$}}
; CHECK-LINKED1: @g1 = global i32 0, !attach !0{{$}}
@g1 = global i32 0, !attach !0

; CHECK: @g2 = external global i32, !attach !0{{$}}
; CHECK: @g3 = weak global i32 1, !attach !0{{$}}
; CHECK-LINKED1: @g2 = global i32 1, !attach !1{{$}}
@g2 = external global i32, !attach !0

; CHECK-LINKED1: @g3 = global i32 2, !attach !1{{$}}
@g3 = weak global i32 1, !attach !0

; CHECK-LINKED2: @g1 = global i32 0, !attach !0{{$}}
; CHECK-LINKED2: @g2 = global i32 1, !attach !1{{$}}
; CHECK-LINKED2: @g3 = global i32 2, !attach !1{{$}}

; CHECK: define void @f1() !attach !0 {
; CHECK-LINKED1: define void @f1() !attach !0 {
define void @f1() !attach !0 {
  call void @f2()
  store i32 0, i32* @g2
  ret void
}

; CHECK: declare !attach !0 void @f2()
; CHECK-LINKED1: define void @f2() !attach !1 {
declare !attach !0 void @f2()

; CHECK: define weak void @f3() !attach !0 {
; CHECK-LINKED1: define void @f3() !attach !1 {
define weak void @f3() !attach !0 {
  ret void
}

; CHECK-LINKED2: define void @f2() !attach !1 {
; CHECK-LINKED2: define void @f3() !attach !1 {
; CHECK-LINKED2: define void @f1() !attach !0 {

; CHECK-LINKED1: !0 = !{i32 0}
; CHECK-LINKED1: !1 = !{i32 1}

; CHECK-LINKED2: !0 = !{i32 0}
; CHECK-LINKED2: !1 = !{i32 1}

!0 = !{i32 0}
