; Test target-specific stack cookie location.
; RUN: llc -mtriple=i386-linux < %s -o - | FileCheck --check-prefix=I386-TLS %s
; RUN: llc -mtriple=x86_64-linux < %s -o - | FileCheck --check-prefix=X64-TLS %s

; RUN: llc -mtriple=i386-linux-android < %s -o - | FileCheck --check-prefix=I386 %s
; RUN: llc -mtriple=i386-linux-android16 < %s -o - | FileCheck --check-prefix=I386 %s
; RUN: llc -mtriple=i386-linux-android17 < %s -o - | FileCheck --check-prefix=I386-TLS %s
; RUN: llc -mtriple=i386-linux-android24 < %s -o - | FileCheck --check-prefix=I386-TLS %s
; RUN: llc -mtriple=x86_64-linux-android < %s -o - | FileCheck --check-prefix=X64-TLS %s
; RUN: llc -mtriple=x86_64-linux-android17 < %s -o - | FileCheck --check-prefix=X64-TLS %s
; RUN: llc -mtriple=x86_64-linux-android24 < %s -o - | FileCheck --check-prefix=X64-TLS %s

; RUN: llc -mtriple=i386-kfreebsd < %s -o - | FileCheck --check-prefix=I386-TLS %s
; RUN: llc -mtriple=x86_64-kfreebsd < %s -o - | FileCheck --check-prefix=X64-TLS %s

define void @_Z1fv() sspreq {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @_Z7CapturePi(i32* nonnull %x)
  ret void
}

declare void @_Z7CapturePi(i32*)

; X64-TLS: movq %fs:40, %[[B:.*]]
; X64-TLS: movq %[[B]], 16(%rsp)
; X64-TLS: movq %fs:40, %[[C:.*]]
; X64-TLS: cmpq 16(%rsp), %[[C]]

; I386: movl __stack_chk_guard, %[[B:.*]]
; I386: movl %[[B]], 8(%esp)
; I386: movl __stack_chk_guard, %[[C:.*]]
; I386: cmpl 8(%esp), %[[C]]

; I386-TLS: movl %gs:20, %[[B:.*]]
; I386-TLS: movl %[[B]], 8(%esp)
; I386-TLS: movl %gs:20, %[[C:.*]]
; I386-TLS: cmpl 8(%esp), %[[C]]
