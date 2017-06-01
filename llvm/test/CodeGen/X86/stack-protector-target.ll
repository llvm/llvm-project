; Test target-specific stack cookie location.
; RUN: llc -mtriple=i386-linux < %s -o - | FileCheck --check-prefix=LINUX-I386 %s
; RUN: llc -mtriple=x86_64-linux < %s -o - | FileCheck --check-prefix=LINUX-X64 %s
; RUN: llc -mtriple=i386-linux-android < %s -o - | FileCheck --check-prefix=LINUX-I386 %s
; RUN: llc -mtriple=x86_64-linux-android < %s -o - | FileCheck --check-prefix=LINUX-X64 %s
; RUN: llc -mtriple=i386-kfreebsd < %s -o - | FileCheck --check-prefix=LINUX-I386 %s
; RUN: llc -mtriple=x86_64-kfreebsd < %s -o - | FileCheck --check-prefix=LINUX-X64 %s

define void @_Z1fv() sspreq {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @_Z7CapturePi(i32* nonnull %x)
  ret void
}

declare void @_Z7CapturePi(i32*)

; LINUX-X64: movq %fs:40, %[[B:.*]]
; LINUX-X64: movq %[[B]], 16(%rsp)
; LINUX-X64: movq %fs:40, %[[C:.*]]
; LINUX-X64: cmpq 16(%rsp), %[[C]]

; LINUX-I386: movl %gs:20, %[[B:.*]]
; LINUX-I386: movl %[[B]], 8(%esp)
; LINUX-I386: movl %gs:20, %[[C:.*]]
; LINUX-I386: cmpl 8(%esp), %[[C]]
