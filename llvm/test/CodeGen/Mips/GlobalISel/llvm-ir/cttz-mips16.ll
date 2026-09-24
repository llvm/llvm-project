; RUN: llc -mtriple=mipsel-linux-gnu -mcpu=mips32 -mattr=+mips16 -verify-machineinstrs < %s | FileCheck %s -check-prefixes=MIPS16

define i32 @cttz_i32(i32 %a) {
; MIPS16-LABEL: cttz_i32:
; MIPS16:       # %bb.0: # %entry
; MIPS16-NEXT:    not $2, $4
; MIPS16-NEXT:	  addiu $4, -1	# 16 bit inst
; MIPS16-NEXT:	  and $4, $2
; MIPS16-NEXT:	  srl $2, $4, 1
; MIPS16-NEXT:	  lw $3, $CPI0_0	# 16 bit inst
; MIPS16-NEXT:	  and $3, $2
; MIPS16-NEXT:	  subu $2, $4, $3
; MIPS16-NEXT:	  lw $3, $CPI0_1	# 16 bit inst
; MIPS16-NEXT:	  srl $4, $2, 2
; MIPS16-NEXT:	  and $2, $3
; MIPS16-NEXT:	  and $4, $3
; MIPS16-NEXT:	  addu $2, $2, $4
; MIPS16-NEXT:	  srl $3, $2, 4
; MIPS16-NEXT:	  addu $2, $2, $3
; MIPS16-NEXT:	  lw $3, $CPI0_2	# 16 bit inst
; MIPS16-NEXT:	  and $3, $2
; MIPS16-NEXT:	  lw $2, $CPI0_3	# 16 bit inst
; MIPS16-NEXT:	  mult $3, $2
; MIPS16-NEXT:	  mflo $2
; MIPS16-NEXT:	  srl $2, $2, 24
; MIPS16-NEXT:	  jrc $ra

entry:
  %0 = call i32 @llvm.cttz.i32(i32 %a, i1 false)
  ret i32 %0
}
declare i32 @llvm.cttz.i32(i32, i1 immarg)

define i64 @cttz_i64(i64  %a) {
; MIPS16-LABEL: cttz_i64:
; MIPS16:       # %bb.1: # %entry
; MIPS16-NEXT:    not $4, $5
; MIPS16-NEXT:	  addiu	$5, -1	# 16 bit inst
; MIPS16-NEXT:	  and $5, $4
; MIPS16-NEXT:	  srl $4, $5, 1
; MIPS16-NEXT:	  and $4, $7
; MIPS16-NEXT:	  subu $4, $5, $4
; MIPS16-NEXT:	  srl $5, $4, 2
; MIPS16-NEXT:	  and $4, $6
; MIPS16-NEXT:	  and $5, $6
; MIPS16-NEXT:	  addu $4, $4, $5
; MIPS16-NEXT:	  srl $5, $4, 4
; MIPS16-NEXT:	  addu $4, $4, $5
; MIPS16-NEXT:	  and $4, $3
; MIPS16-NEXT:	  mult $4, $2
; MIPS16-NEXT:	  mflo $2
; MIPS16-NEXT:	  srl $2, $2, 24
; MIPS16-NEXT:	  addiu	$2, 32	# 16 bit inst
; MIPS16-NEXT:	  li $3, 0
; MIPS16-NEXT:	  jrc $ra

entry:
  %0 = call i64 @llvm.cttz.i64(i64 %a, i1 false)
  ret i64 %0
}
declare i64 @llvm.cttz.i64(i64, i1 immarg)
