; RUN: llc -mtriple=powerpc64-unknown-linux-gnu -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=powerpc-unknown-linux-gnu -verify-machineinstrs < %s | FileCheck %s

; RUN: llc -mtriple=powerpc64-unknown-freebsd -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=powerpc-unknown-freebsd -verify-machineinstrs < %s | FileCheck %s

define dso_local void @moveToVRSave(i32 signext %i) {
entry:
  tail call void asm sideeffect "mtvrsave $0", "r,~{vrsave}"(i32 %i)
  ret void
}

define dso_local signext i32 @moveFromVRSave() {
entry:
  %0 = tail call i32 asm sideeffect "mfvrsave $0", "=r"()
  ret i32 %0
}

define dso_local void @moveToSPR(i32 signext %i) {
entry:
  tail call void asm sideeffect "mtspr 256, $0", "r,~{vrsave}"(i32 %i)
  ret void
}

define dso_local signext i32 @moveFromSPR() {
entry:
  %0 = tail call i32 asm sideeffect "mfspr $0, 256", "=r"()
  ret i32 %0
}

; CHECK-LABEl: moveToVRSave:
; CHECK:         mtvrsave 3

; CHECK-LABEL: moveFromVRSave:
; CHECK:         mfvrsave {{[0-9]+}}

; CHECK-LABEL: moveToSPR:
; CHECK:         mtspr 256, 3

; CHECK-LABEL: moveFromSPR:
; CHECK:         mfspr {{[0-9]}}, 256
