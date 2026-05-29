; RUN: llc -mtriple=mips64el -mcpu=mips64r2 < %s | FileCheck %s -check-prefix=MIPS64

; Test that when a user declares $28 as a global register, assignments to it are
; not optimizes away.
; Clang previously generated no instruction, which GCC generated `move $gp, $4`.
; This test ensures Clang matches GCC behavior.

define void @setglobal_$28(i64 %x) nounwind {
; MIPS64-LABEL: setglobal_$28:
; MIPS64:       # %bb.0: # %entry
; MIPS64-NEXT:    jr $ra
; MIPS64-NEXT:    move $gp, $4

entry:
  tail call void @llvm.write_register.i64(metadata !0, i64 %x)
  ret void
}

define void @setglobal_$gp(i64 %x) nounwind {
; MIPS64-LABEL: setglobal_$gp:
; MIPS64:       # %bb.0: # %entry
; MIPS64-NEXT:    jr $ra
; MIPS64-NEXT:    move $gp, $4

entry:
  tail call void @llvm.write_register.i64(metadata !1, i64 %x)
  ret void
}

declare void @llvm.write_register.i64(metadata, i64) nounwind

!llvm.named.register.$28 = !{!0}
!llvm.named.register.$gp = !{!1}
!0 = !{!"$28"}
!1 = !{!"$gp"}

