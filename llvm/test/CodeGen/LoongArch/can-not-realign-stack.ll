; REQUIRES: expensive_checks
; RUN: llc --mtriple=loongarch64 --frame-pointer=none --mattr=+lasx < %s

; XFAIL: *

;; FIXME: This test will crash with expensive check. The subsequent patch will
;; address and fix this issue.

%struct.S = type { [64 x i16] }

define dso_local noundef signext i32 @main() nounwind {
entry:
  %s = alloca %struct.S, align 2
  call void @llvm.lifetime.start.p0(i64 128, ptr nonnull %s)
  store <16 x i16> <i16 16384, i16 16129, i16 15874, i16 15619, i16 15364, i16 15109, i16 14854, i16 14599, i16 14344, i16 14089, i16 13834, i16 13579, i16 13324, i16 13069, i16 12814, i16 12559>, ptr %s, align 2
  %0 = getelementptr inbounds [64 x i16], ptr %s, i64 0, i64 16
  store <16 x i16> <i16 12304, i16 12049, i16 11794, i16 11539, i16 11284, i16 11029, i16 10774, i16 10519, i16 10264, i16 10009, i16 9754, i16 9499, i16 9244, i16 8989, i16 8734, i16 8479>, ptr %0, align 2
  %1 = getelementptr inbounds [64 x i16], ptr %s, i64 0, i64 32
  store <16 x i16> <i16 8224, i16 7969, i16 7714, i16 7459, i16 7204, i16 6949, i16 6694, i16 6439, i16 6184, i16 5929, i16 5674, i16 5419, i16 5164, i16 4909, i16 4654, i16 4399>, ptr %1, align 2
  %2 = getelementptr inbounds [64 x i16], ptr %s, i64 0, i64 48
  store <16 x i16> <i16 4144, i16 3889, i16 3634, i16 3379, i16 3124, i16 2869, i16 2614, i16 2359, i16 2104, i16 1849, i16 1594, i16 1339, i16 1084, i16 829, i16 574, i16 319>, ptr %2, align 2
  call void @foo(ptr noundef nonnull %s)
  store <16 x i16> <i16 16384, i16 16129, i16 15874, i16 15619, i16 15364, i16 15109, i16 14854, i16 14599, i16 14344, i16 14089, i16 13834, i16 13579, i16 13324, i16 13069, i16 12814, i16 12559>, ptr %s, align 2
  %3 = getelementptr inbounds [64 x i16], ptr %s, i64 0, i64 16
  store <16 x i16> <i16 12304, i16 12049, i16 11794, i16 11539, i16 11284, i16 11029, i16 10774, i16 10519, i16 10264, i16 10009, i16 9754, i16 9499, i16 9244, i16 8989, i16 8734, i16 8479>, ptr %3, align 2
  %4 = getelementptr inbounds [64 x i16], ptr %s, i64 0, i64 32
  store <16 x i16> <i16 8224, i16 7969, i16 7714, i16 7459, i16 7204, i16 6949, i16 6694, i16 6439, i16 6184, i16 5929, i16 5674, i16 5419, i16 5164, i16 4909, i16 4654, i16 4399>, ptr %4, align 2
  %5 = getelementptr inbounds [64 x i16], ptr %s, i64 0, i64 48
  store <16 x i16> <i16 4144, i16 3889, i16 3634, i16 3379, i16 3124, i16 2869, i16 2614, i16 2359, i16 2104, i16 1849, i16 1594, i16 1339, i16 1084, i16 829, i16 574, i16 319>, ptr %5, align 2
  call void @bar(ptr noundef nonnull %s)
  call void @llvm.lifetime.end.p0(i64 128, ptr nonnull %s)
  ret i32 0
}

declare void @foo(ptr nocapture noundef)
declare void @bar(ptr nocapture noundef)

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
