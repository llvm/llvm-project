; RUN: llc -mtriple=i386-linux < %s -o - | FileCheck --check-prefix=LINUX-I386 %s
; RUN: llc -mtriple=x86_64-linux < %s -o - | FileCheck --check-prefix=LINUX-X64 %s
; RUN: llc -mtriple=i386-linux-android < %s -o - | FileCheck --check-prefix=ANDROID-I386 %s
; RUN: llc -mtriple=x86_64-linux-android < %s -o - | FileCheck --check-prefix=ANDROID-X64 %s
; RUN: llc -mtriple=x86_64-fuchsia < %s -o - | FileCheck --check-prefix=FUCHSIA-X64 %s

; RUN: llc -mtriple=i386-linux -safestack-use-pointer-address < %s -o - | FileCheck --check-prefix=LINUX-I386-PA %s

define void @_Z1fv() safestack {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @_Z7CapturePi(i32* nonnull %x)
  ret void
}

declare void @_Z7CapturePi(i32*)

; LINUX-X64: movq __safestack_unsafe_stack_ptr@GOTTPOFF(%rip), %[[A:.*]]
; LINUX-X64: movq %fs:(%[[A]]), %[[B:.*]]
; LINUX-X64: leaq -16(%[[B]]), %[[C:.*]]
; LINUX-X64: movq %[[C]], %fs:(%[[A]])

; LINUX-I386: movl __safestack_unsafe_stack_ptr@INDNTPOFF, %[[A:.*]]
; LINUX-I386: movl %gs:(%[[A]]), %[[B:.*]]
; LINUX-I386: leal -16(%[[B]]), %[[C:.*]]
; LINUX-I386: movl %[[C]], %gs:(%[[A]])

; ANDROID-I386: movl %gs:36, %[[A:.*]]
; ANDROID-I386: leal -16(%[[A]]), %[[B:.*]]
; ANDROID-I386: movl %[[B]], %gs:36

; ANDROID-X64: movq %fs:72, %[[A:.*]]
; ANDROID-X64: leaq -16(%[[A]]), %[[B:.*]]
; ANDROID-X64: movq %[[B]], %fs:72

; FUCHSIA-X64: movq %fs:24, %[[A:.*]]
; FUCHSIA-X64: leaq -16(%[[A]]), %[[B:.*]]
; FUCHSIA-X64: movq %[[B]], %fs:24

; LINUX-I386-PA: calll __safestack_pointer_address
; LINUX-I386-PA: movl %eax, %[[A:.*]]
; LINUX-I386-PA: movl (%eax), %[[B:.*]]
; LINUX-I386-PA: leal -16(%[[B]]), %[[C:.*]]
; LINUX-I386-PA: movl %[[C]], (%[[A]])
