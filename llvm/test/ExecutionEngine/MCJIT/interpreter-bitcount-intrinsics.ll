; RUN: %lli -force-interpreter %s

; The interpreter lowers these intrinsics with IntrinsicLowering, which builds
; the ctpop masks from 64-bit constants. Check types narrower than, not a
; multiple of, and wider than 64 bits. The operands are arguments so that the
; lowered code is executed rather than constant folded. Each failing check
; sets one bit of the exit code.

define i32 @main() {
  %r = call i32 @check(i8 181, i16 -4081, i32 -559038737, i33 4294967297, i64 -1, i128 -1, i8 1, i32 256)
  ret i32 %r
}

define i32 @check(i8 %a, i16 %b, i32 %c, i33 %d, i64 %e, i128 %f, i8 %g, i32 %h) {
  %p8 = call i8 @llvm.ctpop.i8(i8 %a)
  %c8 = icmp ne i8 %p8, 5
  %e0 = zext i1 %c8 to i32

  %p16 = call i16 @llvm.ctpop.i16(i16 %b)
  %c16 = icmp ne i16 %p16, 8
  %z16 = zext i1 %c16 to i32
  %s16 = shl i32 %z16, 1
  %e1 = or i32 %e0, %s16

  %p32 = call i32 @llvm.ctpop.i32(i32 %c)
  %c32 = icmp ne i32 %p32, 24
  %z32 = zext i1 %c32 to i32
  %s32 = shl i32 %z32, 2
  %e2 = or i32 %e1, %s32

  %p33 = call i33 @llvm.ctpop.i33(i33 %d)
  %c33 = icmp ne i33 %p33, 2
  %z33 = zext i1 %c33 to i32
  %s33 = shl i32 %z33, 3
  %e3 = or i32 %e2, %s33

  %p64 = call i64 @llvm.ctpop.i64(i64 %e)
  %c64 = icmp ne i64 %p64, 64
  %z64 = zext i1 %c64 to i32
  %s64 = shl i32 %z64, 4
  %e4 = or i32 %e3, %s64

  %p128 = call i128 @llvm.ctpop.i128(i128 %f)
  %c128 = icmp ne i128 %p128, 128
  %z128 = zext i1 %c128 to i32
  %s128 = shl i32 %z128, 5
  %e5 = or i32 %e4, %s128

  %lz8 = call i8 @llvm.ctlz.i8(i8 %g, i1 false)
  %clz = icmp ne i8 %lz8, 7
  %zlz = zext i1 %clz to i32
  %slz = shl i32 %zlz, 6
  %e6 = or i32 %e5, %slz

  %tz32 = call i32 @llvm.cttz.i32(i32 %h, i1 false)
  %ctz = icmp ne i32 %tz32, 8
  %ztz = zext i1 %ctz to i32
  %stz = shl i32 %ztz, 7
  %e7 = or i32 %e6, %stz

  ret i32 %e7
}
