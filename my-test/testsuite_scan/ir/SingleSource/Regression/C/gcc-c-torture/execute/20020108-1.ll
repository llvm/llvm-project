; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20020108-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20020108-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i8 @ashift_qi_0(i8 noundef returned %0) local_unnamed_addr #0 {
  ret i8 %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 0, -1) i8 @ashift_qi_1(i8 noundef %0) local_unnamed_addr #0 {
  %2 = shl i8 %0, 1
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 0, -3) i8 @ashift_qi_2(i8 noundef %0) local_unnamed_addr #0 {
  %2 = shl i8 %0, 2
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 0, -7) i8 @ashift_qi_3(i8 noundef %0) local_unnamed_addr #0 {
  %2 = shl i8 %0, 3
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 0, -15) i8 @ashift_qi_4(i8 noundef %0) local_unnamed_addr #0 {
  %2 = shl i8 %0, 4
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 0, -31) i8 @ashift_qi_5(i8 noundef %0) local_unnamed_addr #0 {
  %2 = shl i8 %0, 5
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 0, -63) i8 @ashift_qi_6(i8 noundef %0) local_unnamed_addr #0 {
  %2 = shl i8 %0, 6
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 0, -127) i8 @ashift_qi_7(i8 noundef %0) local_unnamed_addr #0 {
  %2 = shl i8 %0, 7
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i8 @lshiftrt_qi_0(i8 noundef returned %0) local_unnamed_addr #0 {
  ret i8 %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 0, -128) i8 @lshiftrt_qi_1(i8 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i8 %0, 1
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 0, 64) i8 @lshiftrt_qi_2(i8 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i8 %0, 2
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 0, 32) i8 @lshiftrt_qi_3(i8 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i8 %0, 3
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 0, 16) i8 @lshiftrt_qi_4(i8 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i8 %0, 4
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 0, 8) i8 @lshiftrt_qi_5(i8 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i8 %0, 5
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 0, 4) i8 @lshiftrt_qi_6(i8 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i8 %0, 6
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 0, 2) i8 @lshiftrt_qi_7(i8 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i8 %0, 7
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i8 @ashiftrt_qi_0(i8 noundef returned %0) local_unnamed_addr #0 {
  ret i8 %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 -64, 64) i8 @ashiftrt_qi_1(i8 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i8 %0, 1
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 -32, 32) i8 @ashiftrt_qi_2(i8 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i8 %0, 2
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 -16, 16) i8 @ashiftrt_qi_3(i8 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i8 %0, 3
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 -8, 8) i8 @ashiftrt_qi_4(i8 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i8 %0, 4
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 -4, 4) i8 @ashiftrt_qi_5(i8 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i8 %0, 5
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 -2, 2) i8 @ashiftrt_qi_6(i8 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i8 %0, 6
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i8 -1, 1) i8 @ashiftrt_qi_7(i8 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i8 %0, 7
  ret i8 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i16 @ashift_hi_0(i16 noundef returned %0) local_unnamed_addr #0 {
  ret i16 %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, -1) i16 @ashift_hi_1(i16 noundef %0) local_unnamed_addr #0 {
  %2 = shl i16 %0, 1
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, -3) i16 @ashift_hi_2(i16 noundef %0) local_unnamed_addr #0 {
  %2 = shl i16 %0, 2
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, -7) i16 @ashift_hi_3(i16 noundef %0) local_unnamed_addr #0 {
  %2 = shl i16 %0, 3
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, -15) i16 @ashift_hi_4(i16 noundef %0) local_unnamed_addr #0 {
  %2 = shl i16 %0, 4
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, -31) i16 @ashift_hi_5(i16 noundef %0) local_unnamed_addr #0 {
  %2 = shl i16 %0, 5
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, -63) i16 @ashift_hi_6(i16 noundef %0) local_unnamed_addr #0 {
  %2 = shl i16 %0, 6
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, -127) i16 @ashift_hi_7(i16 noundef %0) local_unnamed_addr #0 {
  %2 = shl i16 %0, 7
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, -255) i16 @ashift_hi_8(i16 noundef %0) local_unnamed_addr #0 {
  %2 = shl i16 %0, 8
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, -511) i16 @ashift_hi_9(i16 noundef %0) local_unnamed_addr #0 {
  %2 = shl i16 %0, 9
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, -1023) i16 @ashift_hi_10(i16 noundef %0) local_unnamed_addr #0 {
  %2 = shl i16 %0, 10
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, -2047) i16 @ashift_hi_11(i16 noundef %0) local_unnamed_addr #0 {
  %2 = shl i16 %0, 11
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, -4095) i16 @ashift_hi_12(i16 noundef %0) local_unnamed_addr #0 {
  %2 = shl i16 %0, 12
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, -8191) i16 @ashift_hi_13(i16 noundef %0) local_unnamed_addr #0 {
  %2 = shl i16 %0, 13
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, -16383) i16 @ashift_hi_14(i16 noundef %0) local_unnamed_addr #0 {
  %2 = shl i16 %0, 14
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, -32767) i16 @ashift_hi_15(i16 noundef %0) local_unnamed_addr #0 {
  %2 = shl i16 %0, 15
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i16 @lshiftrt_hi_0(i16 noundef returned %0) local_unnamed_addr #0 {
  ret i16 %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, -32768) i16 @lshiftrt_hi_1(i16 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i16 %0, 1
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, 16384) i16 @lshiftrt_hi_2(i16 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i16 %0, 2
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, 8192) i16 @lshiftrt_hi_3(i16 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i16 %0, 3
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, 4096) i16 @lshiftrt_hi_4(i16 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i16 %0, 4
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, 2048) i16 @lshiftrt_hi_5(i16 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i16 %0, 5
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, 1024) i16 @lshiftrt_hi_6(i16 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i16 %0, 6
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, 512) i16 @lshiftrt_hi_7(i16 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i16 %0, 7
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, 256) i16 @lshiftrt_hi_8(i16 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i16 %0, 8
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, 128) i16 @lshiftrt_hi_9(i16 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i16 %0, 9
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, 64) i16 @lshiftrt_hi_10(i16 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i16 %0, 10
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, 32) i16 @lshiftrt_hi_11(i16 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i16 %0, 11
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, 16) i16 @lshiftrt_hi_12(i16 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i16 %0, 12
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, 8) i16 @lshiftrt_hi_13(i16 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i16 %0, 13
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, 4) i16 @lshiftrt_hi_14(i16 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i16 %0, 14
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 0, 2) i16 @lshiftrt_hi_15(i16 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i16 %0, 15
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i16 @ashiftrt_hi_0(i16 noundef returned %0) local_unnamed_addr #0 {
  ret i16 %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 -16384, 16384) i16 @ashiftrt_hi_1(i16 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i16 %0, 1
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 -8192, 8192) i16 @ashiftrt_hi_2(i16 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i16 %0, 2
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 -4096, 4096) i16 @ashiftrt_hi_3(i16 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i16 %0, 3
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 -2048, 2048) i16 @ashiftrt_hi_4(i16 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i16 %0, 4
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 -1024, 1024) i16 @ashiftrt_hi_5(i16 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i16 %0, 5
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 -512, 512) i16 @ashiftrt_hi_6(i16 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i16 %0, 6
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 -256, 256) i16 @ashiftrt_hi_7(i16 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i16 %0, 7
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 -128, 128) i16 @ashiftrt_hi_8(i16 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i16 %0, 8
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 -64, 64) i16 @ashiftrt_hi_9(i16 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i16 %0, 9
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 -32, 32) i16 @ashiftrt_hi_10(i16 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i16 %0, 10
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 -16, 16) i16 @ashiftrt_hi_11(i16 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i16 %0, 11
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 -8, 8) i16 @ashiftrt_hi_12(i16 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i16 %0, 12
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 -4, 4) i16 @ashiftrt_hi_13(i16 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i16 %0, 13
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 -2, 2) i16 @ashiftrt_hi_14(i16 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i16 %0, 14
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i16 -1, 1) i16 @ashiftrt_hi_15(i16 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i16 %0, 15
  ret i16 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @ashift_si_0(i32 noundef returned %0) local_unnamed_addr #0 {
  ret i32 %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -1) i32 @ashift_si_1(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 1
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -3) i32 @ashift_si_2(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 2
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -7) i32 @ashift_si_3(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 3
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -15) i32 @ashift_si_4(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -31) i32 @ashift_si_5(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 5
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -63) i32 @ashift_si_6(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 6
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -127) i32 @ashift_si_7(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 7
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -255) i32 @ashift_si_8(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 8
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -511) i32 @ashift_si_9(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 9
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -1023) i32 @ashift_si_10(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 10
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -2047) i32 @ashift_si_11(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 11
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -4095) i32 @ashift_si_12(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 12
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -8191) i32 @ashift_si_13(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 13
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -16383) i32 @ashift_si_14(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 14
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -32767) i32 @ashift_si_15(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 15
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -65535) i32 @ashift_si_16(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 16
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -131071) i32 @ashift_si_17(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 17
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -262143) i32 @ashift_si_18(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 18
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -524287) i32 @ashift_si_19(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 19
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -1048575) i32 @ashift_si_20(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 20
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -2097151) i32 @ashift_si_21(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 21
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -4194303) i32 @ashift_si_22(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 22
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -8388607) i32 @ashift_si_23(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 23
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -16777215) i32 @ashift_si_24(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 24
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -33554431) i32 @ashift_si_25(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 25
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -67108863) i32 @ashift_si_26(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 26
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -134217727) i32 @ashift_si_27(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 27
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -268435455) i32 @ashift_si_28(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 28
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -536870911) i32 @ashift_si_29(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 29
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -1073741823) i32 @ashift_si_30(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 30
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -2147483647) i32 @ashift_si_31(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 31
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @lshiftrt_si_0(i32 noundef returned %0) local_unnamed_addr #0 {
  ret i32 %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, -2147483648) i32 @lshiftrt_si_1(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 1
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 1073741824) i32 @lshiftrt_si_2(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 2
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 536870912) i32 @lshiftrt_si_3(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 3
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 268435456) i32 @lshiftrt_si_4(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 134217728) i32 @lshiftrt_si_5(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 5
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 67108864) i32 @lshiftrt_si_6(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 6
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 33554432) i32 @lshiftrt_si_7(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 7
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 16777216) i32 @lshiftrt_si_8(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 8
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 8388608) i32 @lshiftrt_si_9(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 9
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 4194304) i32 @lshiftrt_si_10(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 10
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 2097152) i32 @lshiftrt_si_11(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 11
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 1048576) i32 @lshiftrt_si_12(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 12
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 524288) i32 @lshiftrt_si_13(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 13
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 262144) i32 @lshiftrt_si_14(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 14
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 131072) i32 @lshiftrt_si_15(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 15
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 65536) i32 @lshiftrt_si_16(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 16
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 32768) i32 @lshiftrt_si_17(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 17
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 16384) i32 @lshiftrt_si_18(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 18
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 8192) i32 @lshiftrt_si_19(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 19
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 4096) i32 @lshiftrt_si_20(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 20
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 2048) i32 @lshiftrt_si_21(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 21
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 1024) i32 @lshiftrt_si_22(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 22
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 512) i32 @lshiftrt_si_23(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 23
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 256) i32 @lshiftrt_si_24(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 24
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 128) i32 @lshiftrt_si_25(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 25
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 64) i32 @lshiftrt_si_26(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 26
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 32) i32 @lshiftrt_si_27(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 27
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 16) i32 @lshiftrt_si_28(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 28
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 8) i32 @lshiftrt_si_29(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 29
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 4) i32 @lshiftrt_si_30(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 30
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 0, 2) i32 @lshiftrt_si_31(i32 noundef %0) local_unnamed_addr #0 {
  %2 = lshr i32 %0, 31
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @ashiftrt_si_0(i32 noundef returned %0) local_unnamed_addr #0 {
  ret i32 %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -1073741824, 1073741824) i32 @ashiftrt_si_1(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 1
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -536870912, 536870912) i32 @ashiftrt_si_2(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 2
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -268435456, 268435456) i32 @ashiftrt_si_3(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 3
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -134217728, 134217728) i32 @ashiftrt_si_4(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -67108864, 67108864) i32 @ashiftrt_si_5(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 5
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -33554432, 33554432) i32 @ashiftrt_si_6(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 6
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -16777216, 16777216) i32 @ashiftrt_si_7(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 7
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -8388608, 8388608) i32 @ashiftrt_si_8(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 8
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -4194304, 4194304) i32 @ashiftrt_si_9(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 9
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -2097152, 2097152) i32 @ashiftrt_si_10(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 10
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -1048576, 1048576) i32 @ashiftrt_si_11(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 11
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -524288, 524288) i32 @ashiftrt_si_12(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 12
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -262144, 262144) i32 @ashiftrt_si_13(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 13
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -131072, 131072) i32 @ashiftrt_si_14(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 14
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -65536, 65536) i32 @ashiftrt_si_15(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 15
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -32768, 32768) i32 @ashiftrt_si_16(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 16
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -16384, 16384) i32 @ashiftrt_si_17(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 17
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -8192, 8192) i32 @ashiftrt_si_18(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 18
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -4096, 4096) i32 @ashiftrt_si_19(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 19
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -2048, 2048) i32 @ashiftrt_si_20(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 20
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -1024, 1024) i32 @ashiftrt_si_21(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 21
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -512, 512) i32 @ashiftrt_si_22(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 22
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -256, 256) i32 @ashiftrt_si_23(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 23
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -128, 128) i32 @ashiftrt_si_24(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 24
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -64, 64) i32 @ashiftrt_si_25(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 25
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -32, 32) i32 @ashiftrt_si_26(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 26
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -16, 16) i32 @ashiftrt_si_27(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 27
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -8, 8) i32 @ashiftrt_si_28(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 28
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -4, 4) i32 @ashiftrt_si_29(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 29
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -2, 2) i32 @ashiftrt_si_30(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 30
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -1, 1) i32 @ashiftrt_si_31(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 31
  ret i32 %2
}

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  tail call void @exit(i32 noundef 0) #3
  unreachable
}

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
