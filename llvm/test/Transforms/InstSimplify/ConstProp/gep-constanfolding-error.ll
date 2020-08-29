; RUN: opt -gvn -S -o - %s | FileCheck %s
; RUN: opt -newgvn -S -o - %s | FileCheck %s
; Test that the constantfolding getelementptr computation results in
; j[5][4][1] (j+239)
; and not [1][4][4][1] (#449) which is an incorrect out-of-range error
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-none-eabi"

@f = local_unnamed_addr global i32 2, align 4
@t6 = local_unnamed_addr global i32 1, align 4
@j = local_unnamed_addr global [6 x [6 x [7 x i8]]] [[6 x [7 x i8]] [[7 x i8] c"\06\00\00\00\00\00\00", [7 x i8] zeroinitializer, [7 x i8] zeroinitializer, [7 x i8] zeroinitializer, [7 x i8] zeroinitializer, [7 x i8] zeroinitializer], [6 x [7 x i8]] zeroinitializer, [6 x [7 x i8]] zeroinitializer, [6 x [7 x i8]] zeroinitializer, [6 x [7 x i8]] zeroinitializer, [6 x [7 x i8]] zeroinitializer], align 1
@p = internal global i64 0, align 8
@y = local_unnamed_addr global i64* @p, align 4
@b = internal unnamed_addr global i32 0, align 4
@h = common local_unnamed_addr global i16 0, align 2
@a = common local_unnamed_addr global i32 0, align 4
@k = common local_unnamed_addr global i32 0, align 4
@t11 = common local_unnamed_addr global i32 0, align 4

; Function Attrs: nounwind
define i32 @main() local_unnamed_addr {
entry:
  %0 = load i32, i32* @t6, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @t6, align 4
  store i16 4, i16* @h, align 2
  %1 = load i32, i32* @a, align 4
  %conv = trunc i32 %1 to i8
  store i32 1, i32* @f, align 4
  %2 = load i64, i64* @p, align 8
  %cmp4 = icmp slt i64 %2, 2
  %conv6 = zext i1 %cmp4 to i8
  %3 = load i16, i16* @h, align 2
  %conv7 = sext i16 %3 to i32
  %add = add nsw i32 %conv7, 1
  %f.promoted = load i32, i32* @f, align 4
  %4 = mul i32 %conv7, 7
  %5 = add i32 %4, 5
  %6 = sub i32 -1, %f.promoted
  %7 = icmp sgt i32 %6, -2
  %smax = select i1 %7, i32 %6, i32 -2
  %8 = sub i32 6, %smax
  %scevgep = getelementptr [6 x [6 x [7 x i8]]], [6 x [6 x [7 x i8]]]* @j, i32 0, i32 0, i32 %5, i32 %8
  %9 = add i32 %f.promoted, %smax
  %10 = add i32 %9, 2
  call void @llvm.memset.p0i8.i32(i8* %scevgep, i8 %conv6, i32 %10, i1 false)
; CHECK:  call void @llvm.memset.p0i8.i32(i8* getelementptr inbounds ([6 x [6 x [7 x i8]]], [6 x [6 x [7 x i8]]]* @j, i32 0, i{{32|64}} 5, i{{32|64}} 4, i32 1), i8 %conv6, i32 1, i1 false)
; CHECK-NOT: call void @llvm.memset.p0i8.i32(i8* getelementptr ([6 x [6 x [7 x i8]]], [6 x [6 x [7 x i8]]]* @j, i64 1, i64 4, i64 4, i32 1)
  ret i32 0
}
; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1)
