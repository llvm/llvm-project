; RUN: llc < %s -stack-symbol-ordering=0 -mcpu=generic -stackrealign -mattr=+avx -mtriple=x86_64-apple-darwin10 | FileCheck %s
; rdar://11496434
declare void @t1_helper(ptr)
declare void @t3_helper(ptr, ptr)

; Test when forcing stack alignment
define i32 @t8() nounwind uwtable {
entry:
  %a = alloca i32, align 4
  call void @t1_helper(ptr %a) nounwind
  %0 = load i32, ptr %a, align 4
  %add = add nsw i32 %0, 13
  ret i32 %add

; CHECK: _t8
; CHECK:      movq %rsp, %rbp
; CHECK:      andq $-32, %rsp
; CHECK-NEXT: subq $32, %rsp
; CHECK:      movq %rbp, %rsp
; CHECK:      popq %rbp
}

; VLAs
define i32 @t9(i64 %sz) nounwind uwtable {
entry:
  %a = alloca i32, align 4
  %vla = alloca i32, i64 %sz, align 16
  call void @t3_helper(ptr %a, ptr %vla) nounwind
  %0 = load i32, ptr %a, align 4
  %add = add nsw i32 %0, 13
  ret i32 %add

; CHECK: _t9
; CHECK: pushq %rbp
; CHECK: movq %rsp, %rbp
; CHECK: pushq %rbx
; CHECK: andq $-32, %rsp
; CHECK: subq $32, %rsp
; CHECK: movq %rsp, %rbx

; CHECK: leaq -8(%rbp), %rsp
; CHECK: popq %rbx
; CHECK: popq %rbp
}
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"override-stack-alignment", i32 32}
