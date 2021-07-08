; RUN: llc < %s -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -start-before=amdgpu-isel -stop-after=amdgpu-isel -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK

; This caused failure in infinite cycle in Selection DAG (combine) due to missing insert_subvector.
;
; CHECK-LABEL: name: test1
; CHECK: GLOBAL_LOAD_DWORDX4
; CHECK: GLOBAL_LOAD_DWORDX2
; CHECK: GLOBAL_LOAD_DWORDX4
; CHECK: GLOBAL_LOAD_DWORDX2
; CHECK: GLOBAL_STORE_DWORDX4
; CHECK: GLOBAL_STORE_DWORDX2
define protected amdgpu_kernel void @test1(double addrspace(1)* nocapture readonly %srcA, double addrspace(1)* nocapture readonly %srcB, double addrspace(1)* nocapture %dst) local_unnamed_addr #0 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_base_type !7 !kernel_arg_type_qual !8 !kernel_arg_name !9 {
entry:
  %test_step3_double.kernarg.segment = call nonnull align 16 dereferenceable(80) i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr()
  %srcA.kernarg.offset1 = bitcast i8 addrspace(4)* %test_step3_double.kernarg.segment to i8 addrspace(4)*
  %srcA.kernarg.offset.cast = bitcast i8 addrspace(4)* %srcA.kernarg.offset1 to double addrspace(1)* addrspace(4)*
  %0 = bitcast double addrspace(1)* addrspace(4)* %srcA.kernarg.offset.cast to <3 x i64> addrspace(4)*, !amdgpu.uniform !10
  %1 = load <3 x i64>, <3 x i64> addrspace(4)* %0, align 16, !invariant.load !10
  %srcA.load2 = extractelement <3 x i64> %1, i32 0
  %2 = inttoptr i64 %srcA.load2 to double addrspace(1)*
  %srcB.load3 = extractelement <3 x i64> %1, i32 1
  %3 = inttoptr i64 %srcB.load3 to double addrspace(1)*
  %dst.load4 = extractelement <3 x i64> %1, i32 2
  %4 = inttoptr i64 %dst.load4 to double addrspace(1)*
  %srcB.kernarg.offset = getelementptr inbounds i8, i8 addrspace(4)* %test_step3_double.kernarg.segment, i64 8
  %srcB.kernarg.offset.cast = bitcast i8 addrspace(4)* %srcB.kernarg.offset to double addrspace(1)* addrspace(4)*
  %dst.kernarg.offset = getelementptr inbounds i8, i8 addrspace(4)* %test_step3_double.kernarg.segment, i64 16
  %dst.kernarg.offset.cast = bitcast i8 addrspace(4)* %dst.kernarg.offset to double addrspace(1)* addrspace(4)*
  %5 = tail call align 4 dereferenceable(64) i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #5
  %6 = getelementptr i8, i8 addrspace(4)* %5, i64 4
  %7 = bitcast i8 addrspace(4)* %6 to i16 addrspace(4)*, !amdgpu.uniform !10
  %8 = load i16, i16 addrspace(4)* %7, align 4, !range !11, !invariant.load !10
  %9 = tail call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr() #5
  %10 = bitcast i8 addrspace(4)* %9 to i64 addrspace(4)*, !amdgpu.uniform !10
  %11 = load i64, i64 addrspace(4)* %10, align 8, !tbaa !12
  %12 = tail call i32 @llvm.amdgcn.workgroup.id.x() #5
  %13 = tail call i32 @llvm.amdgcn.workitem.id.x() #5, !range !16
  %14 = zext i16 %8 to i32
  %15 = mul i32 %12, %14
  %16 = add i32 %15, %13
  %17 = zext i32 %16 to i64
  %18 = add i64 %11, %17
  %sext = shl i64 %18, 32
  %conv1 = ashr exact i64 %sext, 32
  %19 = mul nsw i64 %conv1, 3
  %20 = getelementptr inbounds double, double addrspace(1)* %2, i64 %19
  %21 = bitcast double addrspace(1)* %20 to <3 x double> addrspace(1)*
  %22 = load <3 x double>, <3 x double> addrspace(1)* %21, align 8, !tbaa !17
  %23 = extractelement <3 x double> %22, i32 0
  %24 = extractelement <3 x double> %22, i32 1
  %25 = extractelement <3 x double> %22, i32 2
  %26 = insertelement <3 x double> undef, double %23, i32 0
  %27 = insertelement <3 x double> %26, double %24, i32 1
  %28 = insertelement <3 x double> %27, double %25, i32 2
  %29 = getelementptr inbounds double, double addrspace(1)* %3, i64 %19
  %30 = bitcast double addrspace(1)* %29 to <3 x double> addrspace(1)*
  %31 = load <3 x double>, <3 x double> addrspace(1)* %30, align 8, !tbaa !17
  %32 = extractelement <3 x double> %31, i32 0
  %33 = extractelement <3 x double> %31, i32 1
  %34 = extractelement <3 x double> %31, i32 2
  %35 = insertelement <3 x double> undef, double %32, i32 0
  %36 = insertelement <3 x double> %35, double %33, i32 1
  %37 = insertelement <3 x double> %36, double %34, i32 2
  %38 = fcmp olt <3 x double> %37, %28
  %39 = select <3 x i1> %38, <3 x double> zeroinitializer, <3 x double> <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %40 = getelementptr inbounds double, double addrspace(1)* %4, i64 %19
  %41 = extractelement <3 x double> %39, i64 0
  %42 = extractelement <3 x double> %39, i64 1
  %43 = insertelement <2 x double> undef, double %41, i32 0
  %44 = insertelement <2 x double> %43, double %42, i32 1
  %45 = bitcast double addrspace(1)* %40 to <2 x double> addrspace(1)*
  store <2 x double> %44, <2 x double> addrspace(1)* %45, align 8, !tbaa !17
  %46 = extractelement <3 x double> %39, i64 2
  %47 = getelementptr inbounds double, double addrspace(1)* %40, i64 2
  store double %46, double addrspace(1)* %47, align 8, !tbaa !17
  ret void
}

; This caused failure in Selection DAG due to lack of insert_subvector implementation.
;
; CHECK-LABEL: name: test2
; CHECK: GLOBAL_LOAD_DWORDX2
; CHECK: GLOBAL_LOAD_DWORDX2
; CHECK: GLOBAL_STORE_DWORDX2
define protected amdgpu_kernel void @test2(double addrspace(1)* nocapture readonly %srcA, double addrspace(1)* nocapture readonly %srcB, double addrspace(1)* nocapture %dst) local_unnamed_addr #0 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_base_type !7 !kernel_arg_type_qual !8 !kernel_arg_name !9 {
entry:
  %test_step3_double.kernarg.segment = call nonnull align 16 dereferenceable(80) i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr()
  %srcA.kernarg.offset1 = bitcast i8 addrspace(4)* %test_step3_double.kernarg.segment to i8 addrspace(4)*
  %srcA.kernarg.offset.cast = bitcast i8 addrspace(4)* %srcA.kernarg.offset1 to double addrspace(1)* addrspace(4)*
  %0 = bitcast double addrspace(1)* addrspace(4)* %srcA.kernarg.offset.cast to <3 x i64> addrspace(4)*, !amdgpu.uniform !10
  %1 = load <3 x i64>, <3 x i64> addrspace(4)* %0, align 16, !invariant.load !10
  %srcA.load2 = extractelement <3 x i64> %1, i32 0
  %2 = inttoptr i64 %srcA.load2 to double addrspace(1)*
  %srcB.load3 = extractelement <3 x i64> %1, i32 1
  %3 = inttoptr i64 %srcB.load3 to double addrspace(1)*
  %dst.load4 = extractelement <3 x i64> %1, i32 2
  %4 = inttoptr i64 %dst.load4 to double addrspace(1)*
  %srcB.kernarg.offset = getelementptr inbounds i8, i8 addrspace(4)* %test_step3_double.kernarg.segment, i64 8
  %srcB.kernarg.offset.cast = bitcast i8 addrspace(4)* %srcB.kernarg.offset to double addrspace(1)* addrspace(4)*
  %dst.kernarg.offset = getelementptr inbounds i8, i8 addrspace(4)* %test_step3_double.kernarg.segment, i64 16
  %dst.kernarg.offset.cast = bitcast i8 addrspace(4)* %dst.kernarg.offset to double addrspace(1)* addrspace(4)*
  %5 = tail call align 4 dereferenceable(64) i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #5
  %6 = getelementptr i8, i8 addrspace(4)* %5, i64 4
  %7 = bitcast i8 addrspace(4)* %6 to i16 addrspace(4)*, !amdgpu.uniform !10
  %8 = load i16, i16 addrspace(4)* %7, align 4, !range !11, !invariant.load !10
  %9 = tail call i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr() #5
  %10 = bitcast i8 addrspace(4)* %9 to i64 addrspace(4)*, !amdgpu.uniform !10
  %11 = load i64, i64 addrspace(4)* %10, align 8, !tbaa !12
  %12 = tail call i32 @llvm.amdgcn.workgroup.id.x() #5
  %13 = tail call i32 @llvm.amdgcn.workitem.id.x() #5, !range !16
  %14 = zext i16 %8 to i32
  %15 = mul i32 %12, %14
  %16 = add i32 %15, %13
  %17 = zext i32 %16 to i64
  %18 = add i64 %11, %17
  %sext = shl i64 %18, 32
  %conv1 = ashr exact i64 %sext, 32
  %19 = mul nsw i64 %conv1, 3
  %20 = getelementptr inbounds double, double addrspace(1)* %2, i64 %19
  %21 = bitcast double addrspace(1)* %20 to <3 x double> addrspace(1)*
  %22 = load <3 x double>, <3 x double> addrspace(1)* %21, align 8, !tbaa !17
  %23 = extractelement <3 x double> %22, i32 0
  %24 = extractelement <3 x double> %22, i32 1
  %25 = extractelement <3 x double> %22, i32 2
  %26 = insertelement <3 x double> undef, double %23, i32 0
  %27 = insertelement <3 x double> %26, double %24, i32 1
  %28 = insertelement <3 x double> %27, double %25, i32 2
  %29 = getelementptr inbounds double, double addrspace(1)* %3, i64 %19
  %30 = bitcast double addrspace(1)* %29 to <3 x double> addrspace(1)*
  %31 = load <3 x double>, <3 x double> addrspace(1)* %30, align 8, !tbaa !17
  %32 = extractelement <3 x double> %31, i32 0
  %33 = extractelement <3 x double> %31, i32 1
  %34 = extractelement <3 x double> %31, i32 2
  %35 = insertelement <3 x double> undef, double %32, i32 0
  %36 = insertelement <3 x double> %35, double %33, i32 1
  %37 = insertelement <3 x double> %36, double %34, i32 2
  %38 = fcmp olt <3 x double> %37, %28
  %39 = select <3 x i1> %38, <3 x double> zeroinitializer, <3 x double> <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>
  %40 = getelementptr inbounds double, double addrspace(1)* %4, i64 %19
  %41 = extractelement <3 x double> %39, i64 0
  %42 = extractelement <3 x double> %39, i64 1
  %43 = insertelement <2 x double> undef, double %41, i32 0
  %44 = insertelement <2 x double> %43, double %42, i32 1
  %45 = bitcast double addrspace(1)* %40 to <2 x double> addrspace(1)*
  %46 = extractelement <3 x double> %39, i64 2
  %47 = getelementptr inbounds double, double addrspace(1)* %40, i64 2
  store double %46, double addrspace(1)* %47, align 8, !tbaa !17
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare i32 @llvm.amdgcn.workgroup.id.x() #1
declare align 4 i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #1
declare align 4 i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr() #1
declare align 4 i8 addrspace(4)* @llvm.amdgcn.kernarg.segment.ptr() #2
declare { i1, i64 } @llvm.amdgcn.if.i64(i1) #3
declare { i1, i64 } @llvm.amdgcn.else.i64.i64(i64) #3
declare i64 @llvm.amdgcn.if.break.i64(i1, i64) #4
declare i1 @llvm.amdgcn.loop.i64(i64) #3
declare void @llvm.amdgcn.end.cf.i64(i64) #3

attributes #0 = { nofree norecurse nosync nounwind willreturn mustprogress "amdgpu-dispatch-ptr" "amdgpu-flat-work-group-size"="1,256" "amdgpu-implicitarg-num-bytes"="56" "amdgpu-implicitarg-ptr" "amdgpu-wave-limiter"="true" "frame-pointer"="none" "min-legal-vector-width"="192" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx900" "target-features"="+16-bit-insts,+ci-insts,+cumode,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64,-xnack" "uniform-work-group-size"="true" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn mustprogress }
attributes #2 = { nounwind readnone speculatable willreturn }
attributes #3 = { convergent nounwind willreturn }
attributes #4 = { convergent nounwind readnone willreturn }
attributes #5 = { nounwind }

!5 = !{i32 1, i32 1, i32 1}
!6 = !{!"none", !"none", !"none"}
!7 = !{!"double*", !"double*", !"double*"}
!8 = !{!"", !"", !""}
!9 = !{!"srcA", !"srcB", !"dst"}
!10 = !{}
!11 = !{i16 1, i16 1025}
!12 = !{!13, !13, i64 0}
!13 = !{!"long", !14, i64 0}
!14 = !{!"omnipotent char", !15, i64 0}
!15 = !{!"Simple C/C++ TBAA"}
!16 = !{i32 0, i32 256}
!17 = !{!18, !18, i64 0}
!18 = !{!"double", !14, i64 0}
