; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20040705-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20040705-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.anon = type { i64, i32 }
%struct.anon.0 = type { i64, i32 }
%struct.anon.1 = type { i64, i32 }

@b = dso_local local_unnamed_addr global %struct.anon zeroinitializer, align 8
@c = dso_local local_unnamed_addr global %struct.anon.0 zeroinitializer, align 8
@d = dso_local local_unnamed_addr global %struct.anon.1 zeroinitializer, align 8

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 64) i32 @ret1() local_unnamed_addr #0 {
  %1 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %2 = and i32 %1, 63
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2048) i32 @ret2() local_unnamed_addr #0 {
  %1 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %2 = lshr i32 %1, 6
  %3 = and i32 %2, 2047
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 32768) i32 @ret3() local_unnamed_addr #0 {
  %1 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %2 = lshr i32 %1, 17
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 32) i32 @ret4() local_unnamed_addr #0 {
  %1 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %2 = and i32 %1, 31
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @ret5() local_unnamed_addr #0 {
  %1 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %2 = lshr i32 %1, 5
  %3 = and i32 %2, 1
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 67108864) i32 @ret6() local_unnamed_addr #0 {
  %1 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %2 = lshr i32 %1, 6
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 65536) i32 @ret7() local_unnamed_addr #0 {
  %1 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %2 = and i32 %1, 65535
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 256) i32 @ret8() local_unnamed_addr #0 {
  %1 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %2 = lshr i32 %1, 16
  %3 = and i32 %2, 255
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 256) i32 @ret9() local_unnamed_addr #0 {
  %1 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %2 = lshr i32 %1, 24
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1_1(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = add i32 %2, %0
  %4 = and i32 %3, 63
  %5 = and i32 %2, -64
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2_1(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = shl i32 %0, 6
  %4 = add i32 %2, %3
  %5 = and i32 %4, 131008
  %6 = and i32 %2, -131009
  %7 = or disjoint i32 %5, %6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn3_1(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = shl i32 %0, 17
  %4 = add i32 %2, %3
  store i32 %4, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn4_1(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = add i32 %2, %0
  %4 = and i32 %3, 31
  %5 = and i32 %2, -32
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn5_1(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = shl i32 %0, 5
  %4 = add i32 %2, %3
  %5 = and i32 %4, 32
  %6 = and i32 %2, -33
  %7 = or disjoint i32 %5, %6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn6_1(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = shl i32 %0, 6
  %4 = add i32 %2, %3
  store i32 %4, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn7_1(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = add i32 %2, %0
  %4 = and i32 %3, 65535
  %5 = and i32 %2, -65536
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn8_1(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = shl i32 %0, 16
  %4 = add i32 %2, %3
  %5 = and i32 %4, 16711680
  %6 = and i32 %2, -16711681
  %7 = or disjoint i32 %5, %6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn9_1(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = shl i32 %0, 24
  %4 = add i32 %2, %3
  store i32 %4, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1_2(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = add i32 %2, 1
  %4 = and i32 %3, 63
  %5 = and i32 %2, -64
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2_2(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = add i32 %2, 64
  %4 = and i32 %3, 131008
  %5 = and i32 %2, -131009
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn3_2(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = and i32 %2, -131072
  %4 = add i32 %3, 131072
  %5 = and i32 %2, 131071
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn4_2(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = add i32 %2, 1
  %4 = and i32 %3, 31
  %5 = and i32 %2, -32
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn5_2(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = xor i32 %2, 32
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn6_2(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = and i32 %2, -64
  %4 = add i32 %3, 64
  %5 = and i32 %2, 63
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn7_2(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = add i32 %2, 1
  %4 = and i32 %3, 65535
  %5 = and i32 %2, -65536
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn8_2(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = add i32 %2, 65536
  %4 = and i32 %3, 16711680
  %5 = and i32 %2, -16711681
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn9_2(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = and i32 %2, -16777216
  %4 = add i32 %3, 16777216
  %5 = and i32 %2, 16777215
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1_3(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = add i32 %2, 1
  %4 = and i32 %3, 63
  %5 = and i32 %2, -64
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2_3(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = add i32 %2, 64
  %4 = and i32 %3, 131008
  %5 = and i32 %2, -131009
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn3_3(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = and i32 %2, -131072
  %4 = add i32 %3, 131072
  %5 = and i32 %2, 131071
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn4_3(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = add i32 %2, 1
  %4 = and i32 %3, 31
  %5 = and i32 %2, -32
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn5_3(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = xor i32 %2, 32
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn6_3(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = and i32 %2, -64
  %4 = add i32 %3, 64
  %5 = and i32 %2, 63
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn7_3(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = add i32 %2, 1
  %4 = and i32 %3, 65535
  %5 = and i32 %2, -65536
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn8_3(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = add i32 %2, 65536
  %4 = and i32 %3, 16711680
  %5 = and i32 %2, -16711681
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn9_3(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = and i32 %2, -16777216
  %4 = add i32 %3, 16777216
  %5 = and i32 %2, 16777215
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1_4(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = sub i32 %2, %0
  %4 = and i32 %3, 63
  %5 = and i32 %2, -64
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2_4(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = shl i32 %0, 6
  %4 = sub i32 %2, %3
  %5 = and i32 %4, 131008
  %6 = and i32 %2, -131009
  %7 = or disjoint i32 %5, %6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn3_4(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = shl i32 %0, 17
  %4 = sub i32 %2, %3
  %5 = and i32 %4, -131072
  %6 = and i32 %2, 131071
  %7 = or disjoint i32 %5, %6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn4_4(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = sub i32 %2, %0
  %4 = and i32 %3, 31
  %5 = and i32 %2, -32
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn5_4(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = shl i32 %0, 5
  %4 = sub i32 %2, %3
  %5 = and i32 %4, 32
  %6 = and i32 %2, -33
  %7 = or disjoint i32 %5, %6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn6_4(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = shl i32 %0, 6
  %4 = sub i32 %2, %3
  %5 = and i32 %4, -64
  %6 = and i32 %2, 63
  %7 = or disjoint i32 %5, %6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn7_4(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = sub i32 %2, %0
  %4 = and i32 %3, 65535
  %5 = and i32 %2, -65536
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn8_4(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = shl i32 %0, 16
  %4 = sub i32 %2, %3
  %5 = and i32 %4, 16711680
  %6 = and i32 %2, -16711681
  %7 = or disjoint i32 %5, %6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn9_4(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = shl i32 %0, 24
  %4 = sub i32 %2, %3
  %5 = and i32 %4, -16777216
  %6 = and i32 %2, 16777215
  %7 = or disjoint i32 %5, %6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1_5(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = add i32 %2, 63
  %4 = and i32 %3, 63
  %5 = and i32 %2, -64
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2_5(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = add i32 %2, 131008
  %4 = and i32 %3, 131008
  %5 = and i32 %2, -131009
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn3_5(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = and i32 %2, -131072
  %4 = add i32 %3, -131072
  %5 = and i32 %2, 131071
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn4_5(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = add i32 %2, 31
  %4 = and i32 %3, 31
  %5 = and i32 %2, -32
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn5_5(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = xor i32 %2, 32
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn6_5(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = and i32 %2, -64
  %4 = add i32 %3, -64
  %5 = and i32 %2, 63
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn7_5(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = add i32 %2, 65535
  %4 = and i32 %3, 65535
  %5 = and i32 %2, -65536
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn8_5(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = add i32 %2, 16711680
  %4 = and i32 %3, 16711680
  %5 = and i32 %2, -16711681
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn9_5(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = and i32 %2, -16777216
  %4 = add i32 %3, -16777216
  %5 = and i32 %2, 16777215
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1_6(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = add i32 %2, 63
  %4 = and i32 %3, 63
  %5 = and i32 %2, -64
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2_6(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = add i32 %2, 131008
  %4 = and i32 %3, 131008
  %5 = and i32 %2, -131009
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn3_6(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = and i32 %2, -131072
  %4 = add i32 %3, -131072
  %5 = and i32 %2, 131071
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn4_6(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = add i32 %2, 31
  %4 = and i32 %3, 31
  %5 = and i32 %2, -32
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn5_6(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = xor i32 %2, 32
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn6_6(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = and i32 %2, -64
  %4 = add i32 %3, -64
  %5 = and i32 %2, 63
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn7_6(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = add i32 %2, 65535
  %4 = and i32 %3, 65535
  %5 = and i32 %2, -65536
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn8_6(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = add i32 %2, 16711680
  %4 = and i32 %3, 16711680
  %5 = and i32 %2, -16711681
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn9_6(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = and i32 %2, -16777216
  %4 = add i32 %3, -16777216
  %5 = and i32 %2, 16777215
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1_7(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = or i32 %0, -64
  %4 = and i32 %3, %2
  store i32 %4, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2_7(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = shl i32 %0, 6
  %4 = or i32 %3, -131009
  %5 = and i32 %4, %2
  store i32 %5, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn3_7(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = shl i32 %0, 17
  %4 = or disjoint i32 %3, 131071
  %5 = and i32 %2, %4
  store i32 %5, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn4_7(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = or i32 %0, -32
  %4 = and i32 %3, %2
  store i32 %4, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn5_7(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = shl i32 %0, 5
  %4 = or i32 %3, -33
  %5 = and i32 %4, %2
  store i32 %5, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn6_7(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = shl i32 %0, 6
  %4 = or disjoint i32 %3, 63
  %5 = and i32 %2, %4
  store i32 %5, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn7_7(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = or i32 %0, -65536
  %4 = and i32 %3, %2
  store i32 %4, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn8_7(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = shl i32 %0, 16
  %4 = or i32 %3, -16711681
  %5 = and i32 %4, %2
  store i32 %5, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn9_7(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = shl i32 %0, 24
  %4 = or disjoint i32 %3, 16777215
  %5 = and i32 %2, %4
  store i32 %5, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1_8(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = and i32 %0, 63
  %4 = or i32 %2, %3
  store i32 %4, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2_8(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = shl i32 %0, 6
  %4 = and i32 %3, 131008
  %5 = or i32 %2, %4
  store i32 %5, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn3_8(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = shl i32 %0, 17
  %4 = or i32 %2, %3
  store i32 %4, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn4_8(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = and i32 %0, 31
  %4 = or i32 %2, %3
  store i32 %4, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn5_8(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = shl i32 %0, 5
  %4 = and i32 %3, 32
  %5 = or i32 %2, %4
  store i32 %5, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn6_8(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = shl i32 %0, 6
  %4 = or i32 %2, %3
  store i32 %4, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn7_8(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = and i32 %0, 65535
  %4 = or i32 %2, %3
  store i32 %4, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn8_8(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = shl i32 %0, 16
  %4 = and i32 %3, 16711680
  %5 = or i32 %2, %4
  store i32 %5, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn9_8(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = shl i32 %0, 24
  %4 = or i32 %2, %3
  store i32 %4, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1_9(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = and i32 %0, 63
  %4 = xor i32 %2, %3
  store i32 %4, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2_9(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = shl i32 %0, 6
  %4 = and i32 %3, 131008
  %5 = xor i32 %2, %4
  store i32 %5, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn3_9(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = shl i32 %0, 17
  %4 = and i32 %2, -131072
  %5 = xor i32 %4, %3
  %6 = and i32 %2, 131071
  %7 = or disjoint i32 %5, %6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn4_9(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = and i32 %0, 31
  %4 = xor i32 %2, %3
  store i32 %4, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn5_9(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = shl i32 %0, 5
  %4 = and i32 %3, 32
  %5 = xor i32 %2, %4
  store i32 %5, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn6_9(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = shl i32 %0, 6
  %4 = and i32 %2, -64
  %5 = xor i32 %4, %3
  %6 = and i32 %2, 63
  %7 = or disjoint i32 %5, %6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn7_9(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = and i32 %0, 65535
  %4 = xor i32 %2, %3
  store i32 %4, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn8_9(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = shl i32 %0, 16
  %4 = and i32 %3, 16711680
  %5 = xor i32 %2, %4
  store i32 %5, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn9_9(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = shl i32 %0, 24
  %4 = and i32 %2, -16777216
  %5 = xor i32 %4, %3
  %6 = and i32 %2, 16777215
  %7 = or disjoint i32 %5, %6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1_a(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = and i32 %2, 63
  %4 = udiv i32 %3, %0
  %5 = and i32 %2, -64
  %6 = or disjoint i32 %5, %4
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2_a(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = lshr i32 %2, 6
  %4 = and i32 %3, 2047
  %5 = udiv i32 %4, %0
  %6 = shl nuw nsw i32 %5, 6
  %7 = and i32 %2, -131009
  %8 = or disjoint i32 %6, %7
  store i32 %8, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn3_a(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = lshr i32 %2, 17
  %4 = udiv i32 %3, %0
  %5 = shl nuw i32 %4, 17
  %6 = and i32 %2, 131071
  %7 = or disjoint i32 %5, %6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn4_a(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = and i32 %2, 31
  %4 = udiv i32 %3, %0
  %5 = and i32 %2, -32
  %6 = or disjoint i32 %5, %4
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn5_a(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = lshr i32 %2, 5
  %4 = and i32 %3, 1
  %5 = udiv i32 %4, %0
  %6 = shl nuw nsw i32 %5, 5
  %7 = and i32 %2, -33
  %8 = or disjoint i32 %6, %7
  store i32 %8, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn6_a(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = lshr i32 %2, 6
  %4 = udiv i32 %3, %0
  %5 = shl nuw i32 %4, 6
  %6 = and i32 %2, 63
  %7 = or disjoint i32 %5, %6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn7_a(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = and i32 %2, 65535
  %4 = udiv i32 %3, %0
  %5 = and i32 %2, -65536
  %6 = or disjoint i32 %5, %4
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn8_a(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = lshr i32 %2, 16
  %4 = and i32 %3, 255
  %5 = udiv i32 %4, %0
  %6 = shl nuw nsw i32 %5, 16
  %7 = and i32 %2, -16711681
  %8 = or disjoint i32 %6, %7
  store i32 %8, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn9_a(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = lshr i32 %2, 24
  %4 = udiv i32 %3, %0
  %5 = shl nuw i32 %4, 24
  %6 = and i32 %2, 16777215
  %7 = or disjoint i32 %5, %6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1_b(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = and i32 %2, 63
  %4 = urem i32 %3, %0
  %5 = and i32 %2, -64
  %6 = or disjoint i32 %5, %4
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2_b(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = lshr i32 %2, 6
  %4 = and i32 %3, 2047
  %5 = urem i32 %4, %0
  %6 = shl nuw nsw i32 %5, 6
  %7 = and i32 %2, -131009
  %8 = or disjoint i32 %6, %7
  store i32 %8, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn3_b(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = lshr i32 %2, 17
  %4 = urem i32 %3, %0
  %5 = shl nuw i32 %4, 17
  %6 = and i32 %2, 131071
  %7 = or disjoint i32 %5, %6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn4_b(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = and i32 %2, 31
  %4 = urem i32 %3, %0
  %5 = and i32 %2, -32
  %6 = or disjoint i32 %5, %4
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn5_b(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = lshr i32 %2, 5
  %4 = and i32 %3, 1
  %5 = urem i32 %4, %0
  %6 = shl nuw nsw i32 %5, 5
  %7 = and i32 %2, -33
  %8 = or disjoint i32 %6, %7
  store i32 %8, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn6_b(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = lshr i32 %2, 6
  %4 = urem i32 %3, %0
  %5 = shl nuw i32 %4, 6
  %6 = and i32 %2, 63
  %7 = or disjoint i32 %5, %6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn7_b(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = and i32 %2, 65535
  %4 = urem i32 %3, %0
  %5 = and i32 %2, -65536
  %6 = or disjoint i32 %5, %4
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn8_b(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = lshr i32 %2, 16
  %4 = and i32 %3, 255
  %5 = urem i32 %4, %0
  %6 = shl nuw nsw i32 %5, 16
  %7 = and i32 %2, -16711681
  %8 = or disjoint i32 %6, %7
  store i32 %8, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn9_b(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = lshr i32 %2, 24
  %4 = urem i32 %3, %0
  %5 = shl nuw i32 %4, 24
  %6 = and i32 %2, 16777215
  %7 = or disjoint i32 %5, %6
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1_c(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = add i32 %2, 3
  %4 = and i32 %3, 63
  %5 = and i32 %2, -64
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2_c(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = add i32 %2, 192
  %4 = and i32 %3, 131008
  %5 = and i32 %2, -131009
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn3_c(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = and i32 %2, -131072
  %4 = add i32 %3, 393216
  %5 = and i32 %2, 131071
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn4_c(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = add i32 %2, 3
  %4 = and i32 %3, 31
  %5 = and i32 %2, -32
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn5_c(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = xor i32 %2, 32
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn6_c(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = and i32 %2, -64
  %4 = add i32 %3, 192
  %5 = and i32 %2, 63
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn7_c(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = add i32 %2, 3
  %4 = and i32 %3, 65535
  %5 = and i32 %2, -65536
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn8_c(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = add i32 %2, 196608
  %4 = and i32 %3, 16711680
  %5 = and i32 %2, -16711681
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn9_c(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = and i32 %2, -16777216
  %4 = add i32 %3, 50331648
  %5 = and i32 %2, 16777215
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1_d(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = add i32 %2, 57
  %4 = and i32 %3, 63
  %5 = and i32 %2, -64
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2_d(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = add i32 %2, 130624
  %4 = and i32 %3, 131008
  %5 = and i32 %2, -131009
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn3_d(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = and i32 %2, -131072
  %4 = add i32 %3, -917504
  %5 = and i32 %2, 131071
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn4_d(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = add i32 %2, 25
  %4 = and i32 %3, 31
  %5 = and i32 %2, -32
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn5_d(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = xor i32 %2, 32
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn6_d(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = and i32 %2, -64
  %4 = add i32 %3, -448
  %5 = and i32 %2, 63
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn7_d(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = add i32 %2, 65529
  %4 = and i32 %3, 65535
  %5 = and i32 %2, -65536
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn8_d(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = add i32 %2, 16318464
  %4 = and i32 %3, 16711680
  %5 = and i32 %2, -16711681
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn9_d(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = and i32 %2, -16777216
  %4 = add i32 %3, -117440512
  %5 = and i32 %2, 16777215
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1_e(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = and i32 %2, -43
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2_e(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = and i32 %2, -129665
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn3_e(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = and i32 %2, 2883583
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn4_e(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = and i32 %2, -11
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @fn5_e(i32 noundef %0) local_unnamed_addr #2 {
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn6_e(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = and i32 %2, 1407
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn7_e(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = and i32 %2, -65515
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn8_e(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = and i32 %2, -15335425
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn9_e(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = and i32 %2, 369098751
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1_f(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = or i32 %2, 19
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2_f(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = or i32 %2, 1216
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn3_f(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = or i32 %2, 2490368
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn4_f(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = or i32 %2, 19
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn5_f(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = or i32 %2, 32
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn6_f(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = or i32 %2, 1216
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn7_f(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = or i32 %2, 19
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn8_f(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = or i32 %2, 1245184
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn9_f(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = or i32 %2, 318767104
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1_g(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = xor i32 %2, 37
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2_g(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = xor i32 %2, 2368
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn3_g(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = xor i32 %2, 4849664
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn4_g(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = xor i32 %2, 5
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn5_g(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = xor i32 %2, 32
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn6_g(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = xor i32 %2, 2368
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn7_g(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = xor i32 %2, 37
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn8_g(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = xor i32 %2, 2424832
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn9_g(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = xor i32 %2, 620756992
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1_h(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = trunc i32 %2 to i8
  %4 = and i8 %3, 63
  %5 = udiv i8 %4, 17
  %6 = zext nneg i8 %5 to i32
  %7 = and i32 %2, -64
  %8 = or disjoint i32 %7, %6
  store i32 %8, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2_h(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = lshr i32 %2, 6
  %4 = trunc i32 %3 to i16
  %5 = and i16 %4, 2047
  %6 = udiv i16 %5, 17
  %7 = shl nuw nsw i16 %6, 6
  %8 = zext nneg i16 %7 to i32
  %9 = and i32 %2, -131009
  %10 = or disjoint i32 %9, %8
  store i32 %10, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn3_h(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = udiv i32 %2, 2228224
  %4 = shl nuw nsw i32 %3, 17
  %5 = and i32 %2, 131071
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn4_h(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = and i32 %2, 31
  %4 = icmp samesign ugt i32 %3, 16
  %5 = zext i1 %4 to i32
  %6 = and i32 %2, -32
  %7 = or disjoint i32 %6, %5
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn5_h(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = and i32 %2, -33
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn6_h(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = udiv i32 %2, 1088
  %4 = shl nuw nsw i32 %3, 6
  %5 = and i32 %2, 63
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn7_h(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = trunc i32 %2 to i16
  %4 = udiv i16 %3, 17
  %5 = zext nneg i16 %4 to i32
  %6 = and i32 %2, -65536
  %7 = or disjoint i32 %6, %5
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn8_h(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = lshr i32 %2, 16
  %4 = trunc i32 %3 to i8
  %5 = udiv i8 %4, 17
  %6 = zext nneg i8 %5 to i32
  %7 = shl nuw nsw i32 %6, 16
  %8 = and i32 %2, -16711681
  %9 = or disjoint i32 %7, %8
  store i32 %9, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn9_h(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = udiv i32 %2, 285212672
  %4 = shl nuw nsw i32 %3, 24
  %5 = and i32 %2, 16777215
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1_i(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = trunc i32 %2 to i8
  %4 = and i8 %3, 63
  %5 = urem i8 %4, 19
  %6 = zext nneg i8 %5 to i32
  %7 = and i32 %2, -64
  %8 = or disjoint i32 %7, %6
  store i32 %8, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2_i(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = lshr i32 %2, 6
  %4 = trunc i32 %3 to i16
  %5 = and i16 %4, 2047
  %6 = urem i16 %5, 19
  %7 = shl nuw nsw i16 %6, 6
  %8 = zext nneg i16 %7 to i32
  %9 = and i32 %2, -131009
  %10 = or disjoint i32 %9, %8
  store i32 %10, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn3_i(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  %3 = urem i32 %2, 2490368
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn4_i(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = freeze i32 %2
  %4 = and i32 %3, 31
  %5 = add nsw i32 %4, -19
  %6 = icmp samesign ult i32 %4, 19
  %7 = select i1 %6, i32 %4, i32 %5
  %8 = and i32 %3, -32
  %9 = or disjoint i32 %7, %8
  store i32 %9, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @fn5_i(i32 noundef %0) local_unnamed_addr #2 {
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn6_i(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  %3 = urem i32 %2, 1216
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn7_i(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = trunc i32 %2 to i16
  %4 = urem i16 %3, 19
  %5 = zext nneg i16 %4 to i32
  %6 = and i32 %2, -65536
  %7 = or disjoint i32 %6, %5
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn8_i(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = lshr i32 %2, 16
  %4 = trunc i32 %3 to i8
  %5 = urem i8 %4, 19
  %6 = zext nneg i8 %5 to i32
  %7 = shl nuw nsw i32 %6, 16
  %8 = and i32 %2, -16711681
  %9 = or disjoint i32 %7, %8
  store i32 %9, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn9_i(i32 noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  %3 = urem i32 %2, 318767104
  store i32 %3, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  store i32 560051, ptr getelementptr inbounds nuw (i8, ptr @b, i64 8), align 8
  store i32 -2147483595, ptr getelementptr inbounds nuw (i8, ptr @c, i64 8), align 8
  store i32 -1147377476, ptr getelementptr inbounds nuw (i8, ptr @d, i64 8), align 8
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
