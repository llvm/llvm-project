; NOTE: Test that -64bit-mode,+64bit-mode,+uintr restores UINTR correctly.
; Regression test for https://github.com/llvm/llvm-project/issues/241375
;
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu \
; RUN:   -mattr=-64bit-mode,+64bit-mode,+uintr | FileCheck %s --check-prefix=RESTORED
;
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu \
; RUN:   -mattr=+uintr | FileCheck %s --check-prefix=BASIC
;
; RUN: not llc < %s -mtriple=i386-unknown-linux-gnu \
; RUN:   -mattr=+uintr 2>&1 | FileCheck %s --check-prefix=REJECT

declare void @llvm.x86.clui()

define void @probe() {
  call void @llvm.x86.clui()
  ret void
}

; RESTORED-LABEL: probe:
; RESTORED:    clui
; RESTORED:    ret

; BASIC-LABEL: probe:
; BASIC:    clui
; BASIC:    ret

; REJECT: error   
