; Check that the MVCLoop (memcpy) is marked as clobbering CC, so that it will
; not be placed betwen two compare and load-on-condition instructions.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -pre-RA-sched=list-ilp \
; RUN:   -print-after=finalize-isel 2>&1 | FileCheck %s
;
; CHECK-LABEL: bb.0.bb:
; CHECK: CLI
; CHECK: LOCGHI
; CHECK-LABEL: bb.2.bb:
; CHECK: MVC

@.str.35 = external dso_local unnamed_addr constant [9 x i8], align 2
@func_38.l_1854 = external dso_local unnamed_addr constant [7 x [10 x [3 x ptr]]], align 8

; Function Attrs: nounwind
define dso_local signext i32 @main(i32 signext %arg, ptr nocapture readonly %arg1) local_unnamed_addr #0 {
bb:
  %tmp = load i8, ptr undef, align 1
  %tmp2 = zext i8 %tmp to i32
  %tmp3 = sub nsw i32 0, %tmp2
  %tmp4 = icmp eq i32 %tmp3, 0
  %tmp5 = zext i1 %tmp4 to i32
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 undef, ptr align 8 @func_38.l_1854, i64 1680, i1 false) #2
  call fastcc void @transparent_crc(i64 undef, ptr @.str.35, i32 signext %tmp5)
  unreachable
}

; Function Attrs: nounwind
declare dso_local fastcc void @transparent_crc(i64, ptr, i32 signext) unnamed_addr #0

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #1

