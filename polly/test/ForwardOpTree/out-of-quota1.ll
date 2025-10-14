; RUN: opt %loadNPMPolly '-passes=print<polly-optree>' -disable-output %s | FileCheck %s

; This used to loop infinitely because of UINT_MAX returned by ISL on out-of-quota.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

%struct.storable_picture.5.221.389.509.629.797.917.1133.1205.1445.1469.1517.1541.1565.1589.1661.1685.1757.1781.1829.1877.2165.2285.2405.2429.2453.2477.2501.2525.2837.2861.2885.2909.4417.1.7.13.17.23.29.55.57.59.61.187 = type { i32, i32, i32, i32, i32, [50 x [6 x [33 x i64]]], [50 x [6 x [33 x i64]]], [50 x [6 x [33 x i64]]], [50 x [6 x [33 x i64]]], i32, i32, i32, i32, i32, i32, i32, i32, i32, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], i32, ptr, i32 }
%struct.DecRefPicMarking_s.4.220.388.508.628.796.916.1132.1204.1444.1468.1516.1540.1564.1588.1660.1684.1756.1780.1828.1876.2164.2284.2404.2428.2452.2476.2500.2524.2836.2860.2884.2908.4416.0.6.12.16.22.28.54.56.58.60.186 = type { i32, i32, i32, i32, i32, ptr }

define void @func() {
entry:
  %0 = load ptr, ptr undef, align 8
  %1 = load ptr, ptr undef, align 8
  %2 = load ptr, ptr undef, align 8
  %3 = load i16, ptr undef, align 4
  %conv2081956 = zext i16 %3 to i64
  br label %for.cond212.preheader

for.cond212.preheader:
  %indvars.iv1926 = phi i64 [ %indvars.iv.next1927, %for.inc354 ], [ 0, %entry ]
  br label %for.body215

for.body215:
  %indvars.iv1921 = phi i64 [ 0, %for.cond212.preheader ], [ %indvars.iv.next1922, %for.body215 ]
  %4 = shl nuw nsw i64 %indvars.iv1921, 1
  %arrayidx230 = getelementptr inbounds %struct.storable_picture.5.221.389.509.629.797.917.1133.1205.1445.1469.1517.1541.1565.1589.1661.1685.1757.1781.1829.1877.2165.2285.2405.2429.2453.2477.2501.2525.2837.2861.2885.2909.4417.1.7.13.17.23.29.55.57.59.61.187, ptr %1, i64 0, i32 5, i64 %indvars.iv1926, i64 1, i64 %4
  store i64 undef, ptr %arrayidx230, align 8
  %5 = or disjoint i64 %4, 1
  %arrayidx248 = getelementptr inbounds %struct.storable_picture.5.221.389.509.629.797.917.1133.1205.1445.1469.1517.1541.1565.1589.1661.1685.1757.1781.1829.1877.2165.2285.2405.2429.2453.2477.2501.2525.2837.2861.2885.2909.4417.1.7.13.17.23.29.55.57.59.61.187, ptr %1, i64 0, i32 5, i64 %indvars.iv1926, i64 1, i64 %5
  store i64 undef, ptr %arrayidx248, align 8
  %arrayidx264 = getelementptr inbounds %struct.storable_picture.5.221.389.509.629.797.917.1133.1205.1445.1469.1517.1541.1565.1589.1661.1685.1757.1781.1829.1877.2165.2285.2405.2429.2453.2477.2501.2525.2837.2861.2885.2909.4417.1.7.13.17.23.29.55.57.59.61.187, ptr %2, i64 0, i32 5, i64 %indvars.iv1926, i64 1, i64 %4
  store i64 undef, ptr %arrayidx264, align 8
  %arrayidx282 = getelementptr inbounds %struct.storable_picture.5.221.389.509.629.797.917.1133.1205.1445.1469.1517.1541.1565.1589.1661.1685.1757.1781.1829.1877.2165.2285.2405.2429.2453.2477.2501.2525.2837.2861.2885.2909.4417.1.7.13.17.23.29.55.57.59.61.187, ptr %2, i64 0, i32 5, i64 %indvars.iv1926, i64 1, i64 %5
  store i64 undef, ptr %arrayidx282, align 8
  %arrayidx298 = getelementptr inbounds %struct.storable_picture.5.221.389.509.629.797.917.1133.1205.1445.1469.1517.1541.1565.1589.1661.1685.1757.1781.1829.1877.2165.2285.2405.2429.2453.2477.2501.2525.2837.2861.2885.2909.4417.1.7.13.17.23.29.55.57.59.61.187, ptr %1, i64 0, i32 5, i64 %indvars.iv1926, i64 0, i64 %4
  store i64 undef, ptr %arrayidx298, align 8
  %arrayidx307 = getelementptr inbounds %struct.storable_picture.5.221.389.509.629.797.917.1133.1205.1445.1469.1517.1541.1565.1589.1661.1685.1757.1781.1829.1877.2165.2285.2405.2429.2453.2477.2501.2525.2837.2861.2885.2909.4417.1.7.13.17.23.29.55.57.59.61.187, ptr %0, i64 0, i32 5, i64 %indvars.iv1926, i64 2, i64 %5
  %6 = load i64, ptr %arrayidx307, align 8
  %arrayidx316 = getelementptr inbounds %struct.storable_picture.5.221.389.509.629.797.917.1133.1205.1445.1469.1517.1541.1565.1589.1661.1685.1757.1781.1829.1877.2165.2285.2405.2429.2453.2477.2501.2525.2837.2861.2885.2909.4417.1.7.13.17.23.29.55.57.59.61.187, ptr %1, i64 0, i32 5, i64 %indvars.iv1926, i64 0, i64 %5
  store i64 %6, ptr %arrayidx316, align 8
  %arrayidx332 = getelementptr inbounds %struct.storable_picture.5.221.389.509.629.797.917.1133.1205.1445.1469.1517.1541.1565.1589.1661.1685.1757.1781.1829.1877.2165.2285.2405.2429.2453.2477.2501.2525.2837.2861.2885.2909.4417.1.7.13.17.23.29.55.57.59.61.187, ptr %2, i64 0, i32 5, i64 %indvars.iv1926, i64 0, i64 %4
  store i64 undef, ptr %arrayidx332, align 8
  %arrayidx350 = getelementptr inbounds %struct.storable_picture.5.221.389.509.629.797.917.1133.1205.1445.1469.1517.1541.1565.1589.1661.1685.1757.1781.1829.1877.2165.2285.2405.2429.2453.2477.2501.2525.2837.2861.2885.2909.4417.1.7.13.17.23.29.55.57.59.61.187, ptr %2, i64 0, i32 5, i64 %indvars.iv1926, i64 0, i64 %5
  store i64 undef, ptr %arrayidx350, align 8
  %indvars.iv.next1922 = add nuw nsw i64 %indvars.iv1921, 1
  %exitcond1925 = icmp eq i64 %indvars.iv.next1922, 16
  br i1 %exitcond1925, label %for.inc354, label %for.body215

for.inc354:
  %indvars.iv.next1927 = add nuw nsw i64 %indvars.iv1926, 1
  %exitcond1930 = icmp eq i64 %indvars.iv1926, %conv2081956
  br i1 %exitcond1930, label %for.body930, label %for.cond212.preheader

for.body930:
  br label %for.body930
}


; CHECK: ForwardOpTree executed, but did not modify anything
