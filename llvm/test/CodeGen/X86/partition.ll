; RUN: llc < %s -mtriple=x86_64-unknown-linux | FileCheck %s

; CHECK: .section .llvm_sympart,"",@llvm_sympart,unique,1
; CHECK-NEXT: .ascii "part1"
; CHECK-NEXT: .zero 1
; CHECK-NEXT: .quad f1
; CHECK-NEXT: .section .llvm_sympart,"",@llvm_sympart,unique,2
; CHECK-NEXT: .ascii "part4"
; CHECK-NEXT: .zero 1
; CHECK-NEXT: .quad g1
; CHECK-NEXT: .section .llvm_sympart,"",@llvm_sympart,unique,3
; CHECK-NEXT: .ascii "part5"
; CHECK-NEXT: .zero 1
; CHECK-NEXT: .quad a1
; CHECK-NEXT: .section .llvm_sympart,"",@llvm_sympart,unique,4
; CHECK-NEXT: .ascii "part6"
; CHECK-NEXT: .zero 1
; CHECK-NEXT: .quad i1

define ptr @f1() partition "part1" {
  unreachable
}

define hidden void @f2() partition "part2" {
  unreachable
}

declare void @f3() partition "part3"

@g1 = global i32 0, partition "part4"

@a1 = alias i32, ptr @g1, partition "part5"
@i1 = ifunc void(), ptr @f1, partition "part6"
