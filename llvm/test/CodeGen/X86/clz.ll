; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s

declare i8 @llvm.cttz.i8(i8, i1)
declare i16 @llvm.cttz.i16(i16, i1)
declare i32 @llvm.cttz.i32(i32, i1)
declare i64 @llvm.cttz.i64(i64, i1)
declare i8 @llvm.ctlz.i8(i8, i1)
declare i16 @llvm.ctlz.i16(i16, i1)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i64 @llvm.ctlz.i64(i64, i1)

define i8 @cttz_i8(i8 %x)  {
; CHECK-LABEL: cttz_i8:
; CHECK:       # BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    bsfl %eax, %eax
; CHECK-NEXT:    retq
  %tmp = call i8 @llvm.cttz.i8( i8 %x, i1 true )
  ret i8 %tmp
}

define i16 @cttz_i16(i16 %x)  {
; CHECK-LABEL: cttz_i16:
; CHECK:       # BB#0:
; CHECK-NEXT:    bsfw %di, %ax
; CHECK-NEXT:    retq
  %tmp = call i16 @llvm.cttz.i16( i16 %x, i1 true )
  ret i16 %tmp
}

define i32 @cttz_i32(i32 %x)  {
; CHECK-LABEL: cttz_i32:
; CHECK:       # BB#0:
; CHECK-NEXT:    bsfl %edi, %eax
; CHECK-NEXT:    retq
  %tmp = call i32 @llvm.cttz.i32( i32 %x, i1 true )
  ret i32 %tmp
}

define i64 @cttz_i64(i64 %x)  {
; CHECK-LABEL: cttz_i64:
; CHECK:       # BB#0:
; CHECK-NEXT:    bsfq %rdi, %rax
; CHECK-NEXT:    retq
  %tmp = call i64 @llvm.cttz.i64( i64 %x, i1 true )
  ret i64 %tmp
}

define i8 @ctlz_i8(i8 %x) {
; CHECK-LABEL: ctlz_i8:
; CHECK:       # BB#0:
; CHECK-NEXT:    movzbl %dil, %eax
; CHECK-NEXT:    bsrl %eax, %eax
; CHECK-NEXT:    xorl $7, %eax
; CHECK-NEXT:    retq
  %tmp2 = call i8 @llvm.ctlz.i8( i8 %x, i1 true )
  ret i8 %tmp2
}

define i16 @ctlz_i16(i16 %x) {
; CHECK-LABEL: ctlz_i16:
; CHECK:       # BB#0:
; CHECK-NEXT:    bsrw %di, %ax
; CHECK-NEXT:    xorl $15, %eax
; CHECK-NEXT:    retq
  %tmp2 = call i16 @llvm.ctlz.i16( i16 %x, i1 true )
  ret i16 %tmp2
}

define i32 @ctlz_i32(i32 %x) {
; CHECK-LABEL: ctlz_i32:
; CHECK:       # BB#0:
; CHECK-NEXT:    bsrl %edi, %eax
; CHECK-NEXT:    xorl $31, %eax
; CHECK-NEXT:    retq
  %tmp = call i32 @llvm.ctlz.i32( i32 %x, i1 true )
  ret i32 %tmp
}

define i64 @ctlz_i64(i64 %x) {
; CHECK-LABEL: ctlz_i64:
; CHECK:       # BB#0:
; CHECK-NEXT:    bsrq %rdi, %rax
; CHECK-NEXT:    xorq $63, %rax
; CHECK-NEXT:    retq
  %tmp = call i64 @llvm.ctlz.i64( i64 %x, i1 true )
  ret i64 %tmp
}

define i32 @ctlz_i32_zero_test(i32 %n) {
; Generate a test and branch to handle zero inputs because bsr/bsf are very slow.

; CHECK-LABEL: ctlz_i32_zero_test:
; CHECK:       # BB#0:
; CHECK-NEXT:    movl $32, %eax
; CHECK-NEXT:    testl %edi, %edi
; CHECK-NEXT:    je .LBB8_2
; CHECK-NEXT:  # BB#1: # %cond.false
; CHECK-NEXT:    bsrl %edi, %eax
; CHECK-NEXT:    xorl $31, %eax
; CHECK-NEXT:  .LBB8_2: # %cond.end
; CHECK-NEXT:    retq
  %tmp1 = call i32 @llvm.ctlz.i32(i32 %n, i1 false)
  ret i32 %tmp1
}

define i32 @ctlz_i32_fold_cmov(i32 %n) {
; Don't generate the cmovne when the source is known non-zero (and bsr would
; not set ZF).
; rdar://9490949
; FIXME: The compare and branch are produced late in IR (by CodeGenPrepare), and
;        codegen doesn't know how to delete the movl and je.

; CHECK-LABEL: ctlz_i32_fold_cmov:
; CHECK:       # BB#0:
; CHECK-NEXT:    orl $1, %edi
; CHECK-NEXT:    movl $32, %eax
; CHECK-NEXT:    je .LBB9_2
; CHECK-NEXT:  # BB#1: # %cond.false
; CHECK-NEXT:    bsrl %edi, %eax
; CHECK-NEXT:    xorl $31, %eax
; CHECK-NEXT:  .LBB9_2: # %cond.end
; CHECK-NEXT:    retq
  %or = or i32 %n, 1
  %tmp1 = call i32 @llvm.ctlz.i32(i32 %or, i1 false)
  ret i32 %tmp1
}

define i32 @ctlz_bsr(i32 %n) {
; Don't generate any xors when a 'ctlz' intrinsic is actually used to compute
; the most significant bit, which is what 'bsr' does natively.

; CHECK-LABEL: ctlz_bsr:
; CHECK:       # BB#0:
; CHECK-NEXT:    bsrl %edi, %eax
; CHECK-NEXT:    retq
  %ctlz = call i32 @llvm.ctlz.i32(i32 %n, i1 true)
  %bsr = xor i32 %ctlz, 31
  ret i32 %bsr
}

define i32 @ctlz_bsr_zero_test(i32 %n) {
; Generate a test and branch to handle zero inputs because bsr/bsf are very slow.
; FIXME: The compare and branch are produced late in IR (by CodeGenPrepare), and
;        codegen doesn't know how to combine the $32 and $31 into $63.

; CHECK-LABEL: ctlz_bsr_zero_test:
; CHECK:       # BB#0:
; CHECK-NEXT:    movl $32, %eax
; CHECK-NEXT:    testl %edi, %edi
; CHECK-NEXT:    je .LBB11_2
; CHECK-NEXT:  # BB#1: # %cond.false
; CHECK-NEXT:    bsrl %edi, %eax
; CHECK-NEXT:    xorl $31, %eax
; CHECK-NEXT:  .LBB11_2: # %cond.end
; CHECK-NEXT:    xorl $31, %eax
; CHECK-NEXT:    retq
  %ctlz = call i32 @llvm.ctlz.i32(i32 %n, i1 false)
  %bsr = xor i32 %ctlz, 31
  ret i32 %bsr
}
