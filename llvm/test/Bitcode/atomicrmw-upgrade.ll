; RUN: llvm-dis < %s.bc | FileCheck %s
; RUN: verify-uselistorder < %s.bc

; atomicrmw-upgrade.ll.bc was produced by running a version of llvm-as from just
; before the IR change on this file.

; CHECK: @atomicrmw
; CHECK:   %b = atomicrmw add ptr %a, i32 %i acquire
define void @atomicrmw(i32* %a, i32 %i) {
    %b = atomicrmw add i32* %a, i32 %i acquire
    ret void
}
