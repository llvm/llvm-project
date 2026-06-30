; RUN: llc -mtriple=mips64el-unknown-linux-gnu -mcpu=mips64r2 -target-abi=n32 < %s -filetype=asm -o - \
; RUN:   | FileCheck  -check-prefixes=N32 %s
; RUN: llc -mtriple=mips64el-unknown-linux-gnu -mcpu=mips64r2 -target-abi=n64 < %s -filetype=asm -o - \
; RUN:   | FileCheck -check-prefixes=N64 %s

; Test that when a user declares $28 as a global register, assignments to it are
; not optimized away.
; Clang previously generated no instruction, which GCC generated `move $gp, $4`.
; This test ensures Clang matches GCC behavior.

define void @setglobal(i64 %x) nounwind {
; N32-LABEL: setglobal:
; N32:       # %bb.0: # %entry
; N32-NEXT:    jr $ra
; N32-NEXT:    move $gp, $4

; N64-LABEL: setglobal:
; N64:       # %bb.0: # %entry
; N64-NEXT:    jr $ra
; N64-NEXT:    move $gp, $4

entry:
  tail call void @llvm.write_register.i64(metadata !0, i64 %x)
  ret void
}

declare void @llvm.write_register.i64(metadata, i64) nounwind

!llvm.named.register.$28 = !{!0}
!0 = !{!"$28"}

