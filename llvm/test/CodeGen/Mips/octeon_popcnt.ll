; RUN: llc -O1 -mtriple=mips64 -mcpu=octeon < %s | FileCheck %s -check-prefix=OCTEON
; RUN: llc -O1 -mtriple=mips64 -mcpu=mips64 < %s | FileCheck %s -check-prefix=MIPS64

define i8 @cnt8(i8 %x) nounwind readnone {
  %cnt = tail call i8 @llvm.ctpop.i8(i8 %x)
  ret i8 %cnt
; OCTEON-LABEL: cnt8:
; OCTEON: jr   $ra
; OCTEON: pop  $2, [[R1:\$[0-9]+]]
; MIPS64-LABEL: cnt8:
; MIPS64-NOT: pop
}

define i16 @cnt16(i16 %x) nounwind readnone {
  %cnt = tail call i16 @llvm.ctpop.i16(i16 %x)
  ret i16 %cnt
; OCTEON-LABEL: cnt16:
; OCTEON: jr   $ra
; OCTEON: pop  $2, [[R1:\$[0-9]+]]
; MIPS64-LABEL: cnt16:
; MIPS64-NOT: pop
}

define i32 @cnt32(i32 signext %x) nounwind readnone {
  %cnt = tail call i32 @llvm.ctpop.i32(i32 %x)
  ret i32 %cnt
; OCTEON-LABEL: cnt32:
; OCTEON: jr   $ra
; OCTEON: pop  $2, $4
; MIPS64-LABEL: cnt32:
; MIPS64-NOT: pop
}

define i64 @cnt64(i64 %x) nounwind readnone {
  %cnt = tail call i64 @llvm.ctpop.i64(i64 %x)
  ret i64 %cnt
; OCTEON-LABEL: cnt64:
; OCTEON: jr   $ra
; OCTEON: dpop $2, $4
; MIPS64-LABEL: cnt64:
; MIPS64-NOT: dpop
}

declare i8 @llvm.ctpop.i8(i8) nounwind readnone
declare i16 @llvm.ctpop.i16(i16) nounwind readnone
declare i32 @llvm.ctpop.i32(i32) nounwind readnone
declare i64 @llvm.ctpop.i64(i64) nounwind readnone
