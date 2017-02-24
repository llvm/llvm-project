; Test target-specific stack cookie location.
; RUN: llc -mtriple=aarch64-linux-android < %s -o - | FileCheck --check-prefix=ANDROID-AARCH64 %s
; RUN: llc -mtriple=aarch64-fuchsia < %s -o - | FileCheck --check-prefix=FUCHSIA-AARCH64 %s

define void @_Z1fv() sspreq {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @_Z7CapturePi(i32* nonnull %x)
  ret void
}

declare void @_Z7CapturePi(i32*)

; ANDROID-AARCH64: mrs [[A:.*]], TPIDR_EL0
; ANDROID-AARCH64: ldr [[B:.*]], {{\[}}[[A]], #40]
; ANDROID-AARCH64: str [[B]], [sp,
; ANDROID-AARCH64: ldr [[C:.*]], {{\[}}[[A]], #40]
; ANDROID-AARCH64: ldr [[D:.*]], [sp,
; ANDROID-AARCH64: cmp [[C]], [[D]]

; FUCHSIA-AARCH64: mrs [[A:.*]], TPIDR_EL0
; FUCHSIA-AARCH64: ldur [[B:.*]], {{\[}}[[A]], #-16]
; FUCHSIA-AARCH64: str [[B]], [sp,
; FUCHSIA-AARCH64: ldur [[C:.*]], {{\[}}[[A]], #-16]
; FUCHSIA-AARCH64: ldr [[D:.*]], [sp,
; FUCHSIA-AARCH64: cmp [[C]], [[D]]
