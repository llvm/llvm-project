; REQUIRES: x86-registered-target
; RUN: opt -mtriple=x86_64-unknown-linux -S -passes=lowertypetests -lowertypetests-summary-action=import -lowertypetests-read-summary=%S/Inputs/import.yaml %s | llc | FileCheck %s

define void @call(ptr %p) {
  ; CHECK:      movl $__typeid_allones7_global_addr, %eax
  ; CHECK-NEXT: subq %rdi, %rax
  ; CHECK-NEXT: rorq $__typeid_allones7_align, %rax
  ; CHECK-NEXT: cmpq $__typeid_allones7_size_m1@ABS8, %rax
  %x = call i1 @llvm.type.test(ptr %p, metadata !"allones7")
  br i1 %x, label %t, label %f

t:
  call void %p()
  ret void

f:
  ret void
}

