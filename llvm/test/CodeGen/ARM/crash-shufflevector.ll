; RUN: llc < %s -mtriple=armv7

declare void @g(<16 x i8>)
define void @f(<4 x i8> %param1, <4 x i8> %param2) {
   %y1 = shufflevector <4 x i8> %param1, <4 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
   %y2 = shufflevector <4 x i8> %param2, <4 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
   %z = shufflevector <16 x i8> %y1, <16 x i8> %y2, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>
   call void @g(<16 x i8> %z)
   ret void
}
