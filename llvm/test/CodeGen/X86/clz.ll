; RUN: llc < %s -march=x86-64 -mcpu=yonah | FileCheck %s

declare i8 @llvm.cttz.i8(i8, i1)
declare i16 @llvm.cttz.i16(i16, i1)
declare i32 @llvm.cttz.i32(i32, i1)
declare i64 @llvm.cttz.i64(i64, i1)
declare i8 @llvm.ctlz.i8(i8, i1)
declare i16 @llvm.ctlz.i16(i16, i1)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i64 @llvm.ctlz.i64(i64, i1)

define i8 @cttz_i8(i8 %x)  {
  %tmp = call i8 @llvm.cttz.i8( i8 %x, i1 true )
  ret i8 %tmp
; CHECK-LABEL: cttz_i8:
; CHECK: bsfl
; CHECK-NOT: cmov
; CHECK: ret
}

define i16 @cttz_i16(i16 %x)  {
  %tmp = call i16 @llvm.cttz.i16( i16 %x, i1 true )
  ret i16 %tmp
; CHECK-LABEL: cttz_i16:
; CHECK: bsfw
; CHECK-NOT: cmov
; CHECK: ret
}

define i32 @cttz_i32(i32 %x)  {
  %tmp = call i32 @llvm.cttz.i32( i32 %x, i1 true )
  ret i32 %tmp
; CHECK-LABEL: cttz_i32:
; CHECK: bsfl
; CHECK-NOT: cmov
; CHECK: ret
}

define i64 @cttz_i64(i64 %x)  {
  %tmp = call i64 @llvm.cttz.i64( i64 %x, i1 true )
  ret i64 %tmp
; CHECK-LABEL: cttz_i64:
; CHECK: bsfq
; CHECK-NOT: cmov
; CHECK: ret
}

define i8 @ctlz_i8(i8 %x) {
entry:
  %tmp2 = call i8 @llvm.ctlz.i8( i8 %x, i1 true )
  ret i8 %tmp2
; CHECK-LABEL: ctlz_i8:
; CHECK: bsrl
; CHECK-NOT: cmov
; CHECK: xorl $7,
; CHECK: ret
}

define i16 @ctlz_i16(i16 %x) {
entry:
  %tmp2 = call i16 @llvm.ctlz.i16( i16 %x, i1 true )
  ret i16 %tmp2
; CHECK-LABEL: ctlz_i16:
; CHECK: bsrw
; CHECK-NOT: cmov
; CHECK: xorl $15,
; CHECK: ret
}

define i32 @ctlz_i32(i32 %x) {
  %tmp = call i32 @llvm.ctlz.i32( i32 %x, i1 true )
  ret i32 %tmp
; CHECK-LABEL: ctlz_i32:
; CHECK: bsrl
; CHECK-NOT: cmov
; CHECK: xorl $31,
; CHECK: ret
}

define i64 @ctlz_i64(i64 %x) {
  %tmp = call i64 @llvm.ctlz.i64( i64 %x, i1 true )
  ret i64 %tmp
; CHECK-LABEL: ctlz_i64:
; CHECK: bsrq
; CHECK-NOT: cmov
; CHECK: xorq $63,
; CHECK: ret
}

define i32 @ctlz_i32_cmov(i32 %n) {
entry:
; Generate a cmov to handle zero inputs when necessary.
; CHECK-LABEL: ctlz_i32_cmov:
; CHECK: bsrl
; CHECK: cmov
; CHECK: xorl $31,
; CHECK: ret
  %tmp1 = call i32 @llvm.ctlz.i32(i32 %n, i1 false)
  ret i32 %tmp1
}

define i32 @ctlz_i32_fold_cmov(i32 %n) {
entry:
; Don't generate the cmovne when the source is known non-zero (and bsr would
; not set ZF).
; rdar://9490949
; CHECK-LABEL: ctlz_i32_fold_cmov:
; CHECK: bsrl
; CHECK-NOT: cmov
; CHECK: xorl $31,
; CHECK: ret
  %or = or i32 %n, 1
  %tmp1 = call i32 @llvm.ctlz.i32(i32 %or, i1 false)
  ret i32 %tmp1
}

define i32 @ctlz_bsr(i32 %n) {
entry:
; Don't generate any xors when a 'ctlz' intrinsic is actually used to compute
; the most significant bit, which is what 'bsr' does natively.
; CHECK-LABEL: ctlz_bsr:
; CHECK: bsrl
; CHECK-NOT: xorl
; CHECK: ret
  %ctlz = call i32 @llvm.ctlz.i32(i32 %n, i1 true)
  %bsr = xor i32 %ctlz, 31
  ret i32 %bsr
}

define i32 @ctlz_bsr_cmov(i32 %n) {
entry:
; Same as ctlz_bsr, but ensure this happens even when there is a potential
; zero.
; CHECK-LABEL: ctlz_bsr_cmov:
; CHECK: bsrl
; CHECK-NOT: xorl
; CHECK: ret
  %ctlz = call i32 @llvm.ctlz.i32(i32 %n, i1 false)
  %bsr = xor i32 %ctlz, 31
  ret i32 %bsr
}
