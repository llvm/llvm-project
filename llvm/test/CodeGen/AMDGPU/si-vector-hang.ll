; RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs | FileCheck %s

; CHECK: {{^}}test_8_min_char:
; CHECK: buffer_store_byte
; CHECK: buffer_store_byte
; CHECK: buffer_store_byte
; CHECK: buffer_store_byte
; CHECK: buffer_store_byte
; CHECK: buffer_store_byte
; CHECK: buffer_store_byte
; CHECK: buffer_store_byte
; ModuleID = 'radeon'

define void @test_8_min_char(i8 addrspace(1)* nocapture %out, i8 addrspace(1)* nocapture readonly %in0, i8 addrspace(1)* nocapture readonly %in1) #0 {
entry:
  %0 = load i8, i8 addrspace(1)* %in0, align 1
  %1 = insertelement <8 x i8> undef, i8 %0, i32 0
  %arrayidx2.i.i = getelementptr inbounds i8, i8 addrspace(1)* %in0, i64 1
  %2 = load i8, i8 addrspace(1)* %arrayidx2.i.i, align 1
  %3 = insertelement <8 x i8> %1, i8 %2, i32 1
  %arrayidx6.i.i = getelementptr inbounds i8, i8 addrspace(1)* %in0, i64 2
  %4 = load i8, i8 addrspace(1)* %arrayidx6.i.i, align 1
  %5 = insertelement <8 x i8> %3, i8 %4, i32 2
  %arrayidx10.i.i = getelementptr inbounds i8, i8 addrspace(1)* %in0, i64 3
  %6 = load i8, i8 addrspace(1)* %arrayidx10.i.i, align 1
  %7 = insertelement <8 x i8> %5, i8 %6, i32 3
  %arrayidx.i.i = getelementptr inbounds i8, i8 addrspace(1)* %in0, i64 4
  %8 = load i8, i8 addrspace(1)* %arrayidx.i.i, align 1
  %9 = insertelement <8 x i8> undef, i8 %8, i32 0
  %arrayidx2.i9.i = getelementptr inbounds i8, i8 addrspace(1)* %in0, i64 5
  %10 = load i8, i8 addrspace(1)* %arrayidx2.i9.i, align 1
  %11 = insertelement <8 x i8> %9, i8 %10, i32 1
  %arrayidx6.i11.i = getelementptr inbounds i8, i8 addrspace(1)* %in0, i64 6
  %12 = load i8, i8 addrspace(1)* %arrayidx6.i11.i, align 1
  %13 = insertelement <8 x i8> %11, i8 %12, i32 2
  %arrayidx10.i13.i = getelementptr inbounds i8, i8 addrspace(1)* %in0, i64 7
  %14 = load i8, i8 addrspace(1)* %arrayidx10.i13.i, align 1
  %15 = insertelement <8 x i8> %13, i8 %14, i32 3
  %vecinit5.i = shufflevector <8 x i8> %7, <8 x i8> %15, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  %16 = load i8, i8 addrspace(1)* %in1, align 1
  %17 = insertelement <8 x i8> undef, i8 %16, i32 0
  %arrayidx2.i.i4 = getelementptr inbounds i8, i8 addrspace(1)* %in1, i64 1
  %18 = load i8, i8 addrspace(1)* %arrayidx2.i.i4, align 1
  %19 = insertelement <8 x i8> %17, i8 %18, i32 1
  %arrayidx6.i.i5 = getelementptr inbounds i8, i8 addrspace(1)* %in1, i64 2
  %20 = load i8, i8 addrspace(1)* %arrayidx6.i.i5, align 1
  %21 = insertelement <8 x i8> %19, i8 %20, i32 2
  %arrayidx10.i.i6 = getelementptr inbounds i8, i8 addrspace(1)* %in1, i64 3
  %22 = load i8, i8 addrspace(1)* %arrayidx10.i.i6, align 1
  %23 = insertelement <8 x i8> %21, i8 %22, i32 3
  %arrayidx.i.i7 = getelementptr inbounds i8, i8 addrspace(1)* %in1, i64 4
  %24 = load i8, i8 addrspace(1)* %arrayidx.i.i7, align 1
  %25 = insertelement <8 x i8> undef, i8 %24, i32 0
  %arrayidx2.i9.i8 = getelementptr inbounds i8, i8 addrspace(1)* %in1, i64 5
  %26 = load i8, i8 addrspace(1)* %arrayidx2.i9.i8, align 1
  %27 = insertelement <8 x i8> %25, i8 %26, i32 1
  %arrayidx6.i11.i9 = getelementptr inbounds i8, i8 addrspace(1)* %in1, i64 6
  %28 = load i8, i8 addrspace(1)* %arrayidx6.i11.i9, align 1
  %29 = insertelement <8 x i8> %27, i8 %28, i32 2
  %arrayidx10.i13.i10 = getelementptr inbounds i8, i8 addrspace(1)* %in1, i64 7
  %30 = load i8, i8 addrspace(1)* %arrayidx10.i13.i10, align 1
  %31 = insertelement <8 x i8> %29, i8 %30, i32 3
  %vecinit5.i11 = shufflevector <8 x i8> %23, <8 x i8> %31, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  %cmp.i = icmp slt <8 x i8> %vecinit5.i, %vecinit5.i11
  %cond.i = select <8 x i1> %cmp.i, <8 x i8> %vecinit5.i, <8 x i8> %vecinit5.i11
  %32 = extractelement <8 x i8> %cond.i, i32 0
  store i8 %32, i8 addrspace(1)* %out, align 1
  %33 = extractelement <8 x i8> %cond.i, i32 1
  %arrayidx2.i.i.i = getelementptr inbounds i8, i8 addrspace(1)* %out, i64 1
  store i8 %33, i8 addrspace(1)* %arrayidx2.i.i.i, align 1
  %34 = extractelement <8 x i8> %cond.i, i32 2
  %arrayidx.i.i.i = getelementptr inbounds i8, i8 addrspace(1)* %out, i64 2
  store i8 %34, i8 addrspace(1)* %arrayidx.i.i.i, align 1
  %35 = extractelement <8 x i8> %cond.i, i32 3
  %arrayidx2.i6.i.i = getelementptr inbounds i8, i8 addrspace(1)* %out, i64 3
  store i8 %35, i8 addrspace(1)* %arrayidx2.i6.i.i, align 1
  %arrayidx.i.i3 = getelementptr inbounds i8, i8 addrspace(1)* %out, i64 4
  %36 = extractelement <8 x i8> %cond.i, i32 4
  store i8 %36, i8 addrspace(1)* %arrayidx.i.i3, align 1
  %37 = extractelement <8 x i8> %cond.i, i32 5
  %arrayidx2.i.i6.i = getelementptr inbounds i8, i8 addrspace(1)* %out, i64 5
  store i8 %37, i8 addrspace(1)* %arrayidx2.i.i6.i, align 1
  %38 = extractelement <8 x i8> %cond.i, i32 6
  %arrayidx.i.i7.i = getelementptr inbounds i8, i8 addrspace(1)* %out, i64 6
  store i8 %38, i8 addrspace(1)* %arrayidx.i.i7.i, align 1
  %39 = extractelement <8 x i8> %cond.i, i32 7
  %arrayidx2.i6.i8.i = getelementptr inbounds i8, i8 addrspace(1)* %out, i64 7
  store i8 %39, i8 addrspace(1)* %arrayidx2.i6.i8.i, align 1
  ret void
}

attributes #0 = { nounwind }

!opencl.kernels = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}

!0 = !{null}
!1 = !{null}
!2 = !{null}
!3 = !{void (i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*)* @test_8_min_char}
!4 = !{null}
!5 = !{null}
!6 = !{null}
!7 = !{null}
!8 = !{null}
