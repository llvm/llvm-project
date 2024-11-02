; RUN: llc -mcpu=pwr9 -O3 -verify-machineinstrs -ppc-vsr-nums-as-vr \
; RUN:     -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:     < %s | FileCheck %s

; RUN: llc -mcpu=pwr9 -O3 -verify-machineinstrs -ppc-vsr-nums-as-vr \
; RUN:     -mtriple=powerpc64-unknown-linux-gnu \
; RUN:     < %s | FileCheck %s --check-prefix=P9BE

; RUN: llc -mcpu=pwr9 -O3 -verify-machineinstrs -ppc-vsr-nums-as-vr \
; RUN:     -mtriple=powerpc64-ibm-aix-xcoff -vec-extabi \
; RUN:     < %s | FileCheck %s --check-prefix=P9BE

; RUN: llc -mcpu=pwr9 -O3 -verify-machineinstrs -ppc-vsr-nums-as-vr \
; RUN:     -mtriple=powerpc-ibm-aix-xcoff -vec-extabi \
; RUN:     < %s | FileCheck %s --check-prefix=P9BE-32

define void @test64(i8* nocapture readonly %pix2, i32 signext %i_pix2) {
; CHECK-LABEL: test64:
; CHECK-NOT: ldux
; CHECK-NOT: mtvsrd
; CHECK: lxsdx [[REG:[0-9]+]]
; CHECK: vperm {{[0-9]+}}, [[REG]]
; P9BE-LABEL: test64:
; P9BE-NOT: ldux
; P9BE-NOT: mtvsrd
; P9BE: lxsdx [[REG:[0-9]+]]
; P9BE: vperm {{[0-9]+}}, {{[0-9]+}}, [[REG]]
; P9BE-32-LABEL: test64:
; P9BE-32: lwzux [[REG1:[0-9]+]]
; P9BE-32: mtfprwz [[REG2:[0-9]+]], [[REG1]]
; P9BE-32: xxinsertw [[REG3:[0-9]+]], [[REG2]]
; P9BE-32: vperm {{[0-9]+}}, {{[0-9]+}}, [[REG3]]
entry:
  %idx.ext63 = sext i32 %i_pix2 to i64
  %add.ptr64 = getelementptr inbounds i8, i8* %pix2, i64 %idx.ext63
  %arrayidx5.1 = getelementptr inbounds i8, i8* %add.ptr64, i64 4
  %0 = bitcast i8* %add.ptr64 to <4 x i16>*
  %1 = load <4 x i16>, <4 x i16>* %0, align 1
  %reorder_shuffle117 = shufflevector <4 x i16> %1, <4 x i16> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  %2 = zext <4 x i16> %reorder_shuffle117 to <4 x i32>
  %3 = sub nsw <4 x i32> zeroinitializer, %2
  %4 = bitcast i8* %arrayidx5.1 to <4 x i16>*
  %5 = load <4 x i16>, <4 x i16>* %4, align 1
  %reorder_shuffle115 = shufflevector <4 x i16> %5, <4 x i16> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  %6 = zext <4 x i16> %reorder_shuffle115 to <4 x i32>
  %7 = sub nsw <4 x i32> zeroinitializer, %6
  %8 = shl nsw <4 x i32> %7, <i32 16, i32 16, i32 16, i32 16>
  %9 = add nsw <4 x i32> %8, %3
  %10 = sub nsw <4 x i32> %9, zeroinitializer
  %11 = shufflevector <4 x i32> undef, <4 x i32> %10, <4 x i32> <i32 2, i32 7, i32 0, i32 5>
  %12 = add nsw <4 x i32> zeroinitializer, %11
  %13 = shufflevector <4 x i32> %12, <4 x i32> undef, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  store <4 x i32> %13, <4 x i32>* undef, align 16
  ret void
}

define void @test32(i8* nocapture readonly %pix2, i32 signext %i_pix2) {
; CHECK-LABEL: test32:
; CHECK-NOT: lwzux
; CHECK-NOT: mtvsrwz
; CHECK: lxsiwzx [[REG:[0-9]+]]
; CHECK: vperm {{[0-9]+}}, [[REG]]
; P9BE-LABEL: test32:
; P9BE-NOT: lwzux
; P9BE-NOT: mtvsrwz
; P9BE: lxsiwzx [[REG:[0-9]+]]
; P9BE: vperm {{[0-9]+}}, {{[0-9]+}}, [[REG]]
; P9BE-32-LABEL: test32:
; P9BE-32: lxsiwzx [[REG:[0-9]+]]
; P9BE-32: vperm {{[0-9]+}}, {{[0-9]+}}, [[REG]]
entry:
  %idx.ext63 = sext i32 %i_pix2 to i64
  %add.ptr64 = getelementptr inbounds i8, i8* %pix2, i64 %idx.ext63
  %arrayidx5.1 = getelementptr inbounds i8, i8* %add.ptr64, i64 4
  %0 = bitcast i8* %add.ptr64 to <4 x i8>*
  %1 = load <4 x i8>, <4 x i8>* %0, align 1
  %reorder_shuffle117 = shufflevector <4 x i8> %1, <4 x i8> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  %2 = zext <4 x i8> %reorder_shuffle117 to <4 x i32>
  %3 = sub nsw <4 x i32> zeroinitializer, %2
  %4 = bitcast i8* %arrayidx5.1 to <4 x i8>*
  %5 = load <4 x i8>, <4 x i8>* %4, align 1
  %reorder_shuffle115 = shufflevector <4 x i8> %5, <4 x i8> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  %6 = zext <4 x i8> %reorder_shuffle115 to <4 x i32>
  %7 = sub nsw <4 x i32> zeroinitializer, %6
  %8 = shl nsw <4 x i32> %7, <i32 16, i32 16, i32 16, i32 16>
  %9 = add nsw <4 x i32> %8, %3
  %10 = sub nsw <4 x i32> %9, zeroinitializer
  %11 = shufflevector <4 x i32> undef, <4 x i32> %10, <4 x i32> <i32 2, i32 7, i32 0, i32 5>
  %12 = add nsw <4 x i32> zeroinitializer, %11
  %13 = shufflevector <4 x i32> %12, <4 x i32> undef, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  store <4 x i32> %13, <4 x i32>* undef, align 16
  ret void
}

define void @test16(i16* nocapture readonly %sums, i32 signext %delta, i32 signext %thresh) {
; CHECK-LABEL: test16
; CHECK-NOT: lhzux
; CHECK: lxsihzx [[REG:[0-9]+]]
; CHECK: vmrghh {{[0-9]+}}, {{[0-9]+}}, [[REG]]
; P9BE-LABEL: test16
; P9BE-NOT: lhzux
; P9BE: lxsihzx [[REG:[0-9]+]]
; P9BE: vperm {{[0-9]+}}, {{[0-9]+}}, [[REG]]
; P9BE-32-LABEL: test16:
; P9BE-32: lhzux [[REG1:[0-9]+]]
; P9BE-32: vmrghh {{[0-9]+}}, {{[0-9]+}}, [[REG1]]
entry:
  %idxprom = sext i32 %delta to i64
  %add14 = add nsw i32 %delta, 8
  %idxprom15 = sext i32 %add14 to i64
  br label %for.body

for.body:                                         ; preds = %entry
  %arrayidx8 = getelementptr inbounds i16, i16* %sums, i64 %idxprom
  %0 = load i16, i16* %arrayidx8, align 2
  %arrayidx16 = getelementptr inbounds i16, i16* %sums, i64 %idxprom15
  %1 = load i16, i16* %arrayidx16, align 2
  %2 = insertelement <4 x i16> undef, i16 %0, i32 2
  %3 = insertelement <4 x i16> %2, i16 %1, i32 3
  %4 = zext <4 x i16> %3 to <4 x i32>
  %5 = sub nsw <4 x i32> zeroinitializer, %4
  %6 = sub nsw <4 x i32> zeroinitializer, %5
  %7 = select <4 x i1> undef, <4 x i32> %6, <4 x i32> %5
  %bin.rdx = add <4 x i32> %7, zeroinitializer
  %rdx.shuf54 = shufflevector <4 x i32> %bin.rdx, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx55 = add <4 x i32> %bin.rdx, %rdx.shuf54
  %8 = extractelement <4 x i32> %bin.rdx55, i32 0
  %op.extra = add nuw i32 %8, 0
  %cmp25 = icmp slt i32 %op.extra, %thresh
  br i1 %cmp25, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  unreachable

if.end:                                           ; preds = %for.body
  ret void
}

define void @test8(i8* nocapture readonly %sums, i32 signext %delta, i32 signext %thresh) {
; CHECK-LABEL: test8:
; CHECK-NOT: lbzux
; CHECK: lxsibzx [[REG:[0-9]+]]
; CHECK: vmrghb {{[0-9]+}}, {{[0-9]+}}, [[REG]]
; P9BE-LABEL: test8:
; P9BE-NOT: lbzux
; P9BE: lxsibzx [[REG:[0-9]+]]
; P9BE: vperm {{[0-9]+}}, {{[0-9]+}}, [[REG]]
; P9BE-32-LABEL: test8:
; P9BE-32: lxsibzx [[REG:[0-9]+]]
; P9BE-32: vperm {{[0-9]+}}, {{[0-9]+}}, [[REG]]
entry:
  %idxprom = sext i32 %delta to i64
  %add14 = add nsw i32 %delta, 8
  %idxprom15 = sext i32 %add14 to i64
  br label %for.body

for.body:                                         ; preds = %entry
  %arrayidx8 = getelementptr inbounds i8, i8* %sums, i64 %idxprom
  %0 = load i8, i8* %arrayidx8, align 2
  %arrayidx16 = getelementptr inbounds i8, i8* %sums, i64 %idxprom15
  %1 = load i8, i8* %arrayidx16, align 2
  %2 = insertelement <4 x i8> undef, i8 %0, i32 2
  %3 = insertelement <4 x i8> %2, i8 %1, i32 3
  %4 = zext <4 x i8> %3 to <4 x i32>
  %5 = sub nsw <4 x i32> zeroinitializer, %4
  %6 = sub nsw <4 x i32> zeroinitializer, %5
  %7 = select <4 x i1> undef, <4 x i32> %6, <4 x i32> %5
  %bin.rdx = add <4 x i32> %7, zeroinitializer
  %rdx.shuf54 = shufflevector <4 x i32> %bin.rdx, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx55 = add <4 x i32> %bin.rdx, %rdx.shuf54
  %8 = extractelement <4 x i32> %bin.rdx55, i32 0
  %op.extra = add nuw i32 %8, 0
  %cmp25 = icmp slt i32 %op.extra, %thresh
  br i1 %cmp25, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  unreachable

if.end:                                           ; preds = %for.body
  ret void
}
