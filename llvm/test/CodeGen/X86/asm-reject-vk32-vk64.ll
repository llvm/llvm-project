; RUN: not llc -o /dev/null %s -mtriple=x86_64-unknown-unknown -mattr=avx512f 2>&1 | FileCheck %s
; RUN: not llc -o /dev/null %s -mtriple=i386-unknown-unknown -mattr=avx512f 2>&1 | FileCheck %s

; CHECK: error: couldn't allocate input reg for constraint 'Yk'
define <8 x i64> @mask_Yk_i32(i32 %msk, <8 x i64> %x, <8 x i64> %y) {
entry:
  %0 = tail call <8 x i64> asm "vpaddw\09$3, $2, $0 {$1}", "=x,^Yk,x,x,~{dirflag},~{fpsr},~{flags}"(i32 %msk, <8 x i64> %x, <8 x i64> %y)
  ret <8 x i64> %0
}

; CHECK: error: couldn't allocate input reg for constraint 'Yk'
define <8 x i64> @mask_Yk_i64(i64 %msk, <8 x i64> %x, <8 x i64> %y) {
entry:
  %0 = tail call <8 x i64> asm "vpaddb\09$3, $2, $0 {$1}", "=x,^Yk,x,x,~{dirflag},~{fpsr},~{flags}"(i64 %msk, <8 x i64> %x, <8 x i64> %y)
  ret <8 x i64> %0
}

; CHECK: error: couldn't allocate output register for constraint 'k'
define i32 @k_wise_op_i32(i32 %msk_src1, i32 %msk_src2) {
entry:
  %0 = tail call i32 asm "kandd\09$2, $1, $0", "=k,k,k,~{dirflag},~{fpsr},~{flags}"(i32 %msk_src1, i32 %msk_src2)
  ret i32 %0
}

; CHECK: error: couldn't allocate output register for constraint 'k'
define i64 @k_wise_op_i64(i64 %msk_src1, i64 %msk_src2) {
entry:
  %0 = tail call i64 asm "kandq\09$2, $1, $0", "=k,k,k,~{dirflag},~{fpsr},~{flags}"(i64 %msk_src1, i64 %msk_src2)
  ret i64 %0
}

