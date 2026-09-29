; Check that custom handling of SETCC does not crash
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13

@g_39 = external global [4 x i8], align 2
@g_2166 = external global <{ i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32 }>, align 8

; Function Attrs: nounwind
define void @main() local_unnamed_addr #0 {
  %1 = load volatile i88, ptr @g_2166, align 8
  %2 = lshr i88 %1, 87
  %3 = trunc i88 %2 to i64
  %4 = icmp sgt i64 %3, 9293
  %5 = zext i1 %4 to i32
  %6 = lshr i32 %5, 0
  %7 = shl i32 %6, 6
  %8 = trunc i32 %7 to i8
  store i8 %8, ptr getelementptr inbounds ([4 x i8], ptr @g_39, i64 0, i64 1), align 1
  unreachable
}
