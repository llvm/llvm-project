; RUN: llc -mtriple=mips64el -mcpu=mips64r2 < %s | FileCheck %s -check-prefix=MIPS64

define void @setglobal(i64 %x) {
; MIPS64-LABEL: setglobal:
; MIPS64:       # %bb.0: # %entry
; MIPS64-NEXT:    jr $ra
; MIPS64-NEXT:    move $gp, $4

entry:
  tail call void @llvm.write_register.i64(metadata !0, i64 %x)
  ret void
}

declare void @llvm.write_register.i64(metadata, i64) #1

!llvm.named.register.$28 = !{!0}
!0 = !{!"$28"}

