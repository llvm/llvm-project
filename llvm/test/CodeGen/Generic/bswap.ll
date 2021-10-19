; tests lowering of vector bswap
; RUN: lli -jit-kind=mcjit -force-interpreter %s | FileCheck %s

; CHECK: 0x100
; CHECK: 0x10000
; CHECK: 0x1001000000000000
; CHECK: 0x100
; CHECK: 0x10000
; CHECK: 0x1001000000000000



target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare i16 @llvm.bswap.i16(i16);
declare i32 @llvm.bswap.i32(i32);
declare i64 @llvm.bswap.i64(i64);
declare <4 x i16> @llvm.bswap.v4i16(<4 x i16>);
declare <4 x i32> @llvm.bswap.v4i32(<4 x i32>);
declare <4 x i64> @llvm.bswap.v4i64(<4 x i64>);
declare i32 @printf(i8* nocapture readonly, ...);

@.str = private unnamed_addr constant [5 x i8] c"%#x\0A\00", align 1
@.strs = private unnamed_addr constant [6 x i8] c"%#hx\0A\00", align 1
@.strl = private unnamed_addr constant [6 x i8] c"%#lx\0A\00", align 1

define i32 @main() local_unnamed_addr {
  %ra = tail call i16 @llvm.bswap.i16(i16 1)
  %pa = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.strs, i64 0, i64 0), i16 %ra)

  %rb = tail call i32 @llvm.bswap.i32(i32 256)
  %pb = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i64 0, i64 0), i32 %rb)

  %rc = tail call i64 @llvm.bswap.i64(i64 272)
  %pc = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.strl, i64 0, i64 0), i64 %rc)

  %r0 = tail call <4 x i16> @llvm.bswap.v4i16(<4 x i16> <i16 1, i16 1, i16 1, i16 1>)
  %e0 = extractelement <4 x i16> %r0, i8 0
  %p0 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.strs, i64 0, i64 0), i16 %e0)

  %r1 = tail call <4 x i32> @llvm.bswap.v4i32(<4 x i32> <i32 256, i32 256, i32 256, i32 256>)
  %e1 = extractelement <4 x i32> %r1, i8 1
  %p1 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i64 0, i64 0), i32 %e1)

  %r2 = tail call <4 x i64> @llvm.bswap.v4i64(<4 x i64> <i64 272, i64 272, i64 272, i64 272>)
  %e2 = extractelement <4 x i64> %r2, i8 2
  %p2 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.strl, i64 0, i64 0), i64 %e2)

  ret i32 0
}
