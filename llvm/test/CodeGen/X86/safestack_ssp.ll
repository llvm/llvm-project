; Test codegen pipeline for SafeStack + StackProtector combination.
; RUN: llc -mtriple=i386-linux < %s -o - | FileCheck --check-prefix=LINUX-I386 %s
; RUN: llc -mtriple=x86_64-linux < %s -o - | FileCheck --check-prefix=LINUX-X64 %s

define void @_Z1fv() safestack sspreq {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @_Z7CapturePi(i32* nonnull %x)
  ret void
}

declare void @_Z7CapturePi(i32*)

; LINUX-X64-DAG: movq __safestack_unsafe_stack_ptr@GOTTPOFF(%rip), %[[A:.*]]
; LINUX-X64-DAG: movq %fs:(%[[A]]), %[[B:.*]]
; LINUX-X64-DAG: movq %fs:40, %[[COOKIE:.*]]
; LINUX-X64-DAG: leaq -16(%[[B]]), %[[C:.*]]
; LINUX-X64-DAG: movq %[[C]], %fs:(%[[A]])
; LINUX-X64-DAG: movq %[[COOKIE]], -8(%[[B]])

; LINUX-I386-DAG: movl __safestack_unsafe_stack_ptr@INDNTPOFF, %[[A:.*]]
; LINUX-I386-DAG: movl %gs:(%[[A]]), %[[B:.*]]
; LINUX-I386-DAG: movl %gs:20, %[[COOKIE:.*]]
; LINUX-I386-DAG: leal -16(%[[B]]), %[[C:.*]]
; LINUX-I386-DAG: movl %[[C]], %gs:(%[[A]])
; LINUX-I386-DAG: movl %[[COOKIE]], -4(%[[B]])
