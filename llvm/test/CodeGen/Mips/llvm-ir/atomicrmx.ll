; RUN: llc -asm-show-inst  -mtriple=mipsel -mcpu=mips32r6 < %s | \
; RUN:    FileCheck %s -check-prefix=CHK32
; RUN: llc -asm-show-inst  -mtriple=mips64el -mcpu=mips64r6 < %s | \
; RUN:    FileCheck %s -check-prefix=CHK64

@a = common global i32 0, align 4
@b = common global i64 0, align 8


define i32 @ll_sc(i32 signext %x) {
; CHK32-LABEL: ll_sc

;CHK32:  LL_R6
;CHK32:  SC_R6
  %1 = atomicrmw add ptr @a, i32 %x monotonic
  ret i32 %1
}

define i64 @lld_scd(i64 signext %x) {
; CHK64-LABEL: lld_scd

;CHK64:  LLD_R6
;CHK64:  SCD_R6
  %1 = atomicrmw add ptr @b, i64 %x monotonic
  ret i64 %1
}
