; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -infer-address-spaces %s | FileCheck %s

; check that the debug locations are correctly propagated

@lds = internal unnamed_addr addrspace(3) global [648 x double] undef, align 8

; CHECK-LABEL: @load_global_from_flat(
; CHECK-NEXT: %tmp0 = addrspacecast float* %generic_scalar to float addrspace(1)*, !dbg ![[DEBUG_LOC_TMP0:[0-9]+]]
; CHECK-NEXT: %tmp1 = load float, float addrspace(1)* %tmp0, align 4, !dbg ![[DEBUG_LOC_TMP1:[0-9]+]]
; CHECK-NEXT: ret float %tmp1, !dbg ![[DEBUG_LOC_RET:[0-9]+]]
define float @load_global_from_flat(float* %generic_scalar) #0 !dbg !5 {
  %tmp0 = addrspacecast float* %generic_scalar to float addrspace(1)*, !dbg !8
  %tmp1 = load float, float addrspace(1)* %tmp0, align 4, !dbg !9
  ret float %tmp1, !dbg !10
}

; CHECK-LABEL: @simplified_constexpr_gep_addrspacecast(
; CHECK: %gep0 = getelementptr inbounds double, double addrspace(3)* getelementptr inbounds ([648 x double], [648 x double] addrspace(3)* @lds, i64 0, i64 384), i64 %idx0, !dbg ![[DEBUG_LOC_GEP0:[0-9]+]]
; CHECK-NEXT: store double 1.000000e+00, double addrspace(3)* %gep0, align 8, !dbg ![[DEBUG_LOC_STORE_GEP0:[0-9]+]]
define void @simplified_constexpr_gep_addrspacecast(i64 %idx0, i64 %idx1) #0 !dbg !11 {
  %gep0 = getelementptr inbounds double, double* addrspacecast (double addrspace(3)* getelementptr inbounds ([648 x double], [648 x double] addrspace(3)* @lds, i64 0, i64 384) to double*), i64 %idx0, !dbg !12
  %asc = addrspacecast double* %gep0 to double addrspace(3)*, !dbg !13
  store double 1.000000e+00, double addrspace(3)* %asc, align 8, !dbg !14
  ret void, !dbg !15
}

; CHECK-LABEL: @objectsize_group_to_flat_i32(
; CHECK: %val = call i32 @llvm.objectsize.i32.p3i8(i8 addrspace(3)* %group.ptr, i1 true, i1 false, i1 false), !dbg ![[DEBUG_LOC_VAL:[0-9]+]]
define i32 @objectsize_group_to_flat_i32(i8 addrspace(3)* %group.ptr) #0 !dbg !16 {
  %cast = addrspacecast i8 addrspace(3)* %group.ptr to i8*, !dbg !17
  %val = call i32 @llvm.objectsize.i32.p0i8(i8* %cast, i1 true, i1 false, i1 false), !dbg !18
  ret i32 %val, !dbg !19
}

; CHECK-LABEL: @memset_group_to_flat(
; CHECK: call void @llvm.memset.p3i8.i64(i8 addrspace(3)* align 4 %group.ptr, i8 4, i64 32, i1 false), !dbg ![[DEBUG_LOC_MEMSET_CAST:[0-9]+]]
define amdgpu_kernel void @memset_group_to_flat(i8 addrspace(3)* %group.ptr, i32 %y) #0 !dbg !20 {
  %cast = addrspacecast i8 addrspace(3)* %group.ptr to i8*, !dbg !21
  call void @llvm.memset.p0i8.i64(i8* align 4 %cast, i8 4, i64 32, i1 false), !dbg !22, !tbaa !23, !alias.scope !26, !noalias !29
  ret void, !dbg !31
}

; CHECK-LABEL: @ptrmask_cast_global_to_flat(
; CHECK-NEXT:    [[PTRMASK:%.*]] = call i8 addrspace(1)* @llvm.ptrmask.p1i8.i64(i8 addrspace(1)* %src.ptr, i64 %mask), !dbg ![[DEBUG_LOC_PTRMASK:[0-9]+]]
; CHECK-NEXT:    %load = load i8, i8 addrspace(1)* [[PTRMASK]], align 1, !dbg ![[DEBUG_LOC_LOAD:[0-9]+]]
define i8 @ptrmask_cast_global_to_flat(i8 addrspace(1)* %src.ptr, i64 %mask) #0 !dbg !32 {
  %cast = addrspacecast i8 addrspace(1)* %src.ptr to i8*, !dbg !33
  %masked = call i8* @llvm.ptrmask.p0i8.i64(i8* %cast, i64 %mask), !dbg !34
  %load = load i8, i8* %masked, !dbg !35
  ret i8 %load, !dbg !36
}

; the new addrspacecast gets the debug location from it user (in this case, the gep)
; CHECK-LABEL: @assume_addresspace(
; CHECK: [[ASCAST:%.*]] = addrspacecast float* %p to float addrspace(3)*, !dbg ![[DEBUG_LOC_ARRAYIDX:[0-9]+]]
; CHECK-NEXT: %arrayidx = getelementptr inbounds float, float addrspace(3)* [[ASCAST]], i64 %x64, !dbg ![[DEBUG_LOC_ARRAYIDX]]
; CHECK-NEXT: %arrayidx.load = load float, float addrspace(3)* %arrayidx, align 4, !dbg ![[DEBUG_LOC_ARRAYIDX_LOAD:[0-9]+]]
define float @assume_addresspace(float* %p) !dbg !37 {
entry:
  %cast = bitcast float* %p to i8*, !dbg !38
  %is_shared = call i1 @llvm.amdgcn.is.shared(i8* %cast), !dbg !39
  tail call void @llvm.assume(i1 %is_shared), !dbg !40
  %x32 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !41
  %x64 = zext i32 %x32 to i64, !dbg !42
  %arrayidx = getelementptr inbounds float, float* %p, i64 %x64, !dbg !43
  %arrayidx.load = load float, float* %arrayidx, align 4, !dbg !44
  ret float %arrayidx.load, !dbg !45
}

declare i32 @llvm.objectsize.i32.p0i8(i8*, i1 immarg, i1 immarg, i1 immarg) #1
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #2
declare i8* @llvm.ptrmask.p0i8.i64(i8*, i64) #3
declare i1 @llvm.amdgcn.is.shared(i8* nocapture) #4
declare i32 @llvm.amdgcn.workitem.id.x() #4
declare void @llvm.assume(i1)

attributes #0 = { nounwind }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { argmemonly nocallback nofree nounwind willreturn writeonly }
attributes #3 = { nounwind readnone speculatable willreturn }
attributes #4 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

; CHECK: ![[DEBUG_LOC_TMP0]] = !DILocation(line: 1, column: 1,
; CHECK: ![[DEBUG_LOC_TMP1]] = !DILocation(line: 2, column: 1,
; CHECK: ![[DEBUG_LOC_RET]] = !DILocation(line: 3, column: 1,
; CHECK: ![[DEBUG_LOC_GEP0]] = !DILocation(line: 4, column: 1,
; CHECK: ![[DEBUG_LOC_STORE_GEP0]] = !DILocation(line: 6, column: 1,
; CHECK: ![[DEBUG_LOC_VAL]] = !DILocation(line: 9, column: 1,
; CHECK: ![[DEBUG_LOC_MEMSET_CAST]] = !DILocation(line: 12, column: 1,
; CHECK: ![[DEBUG_LOC_PTRMASK]] = !DILocation(line: 15, column: 1,
; CHECK: ![[DEBUG_LOC_LOAD]] = !DILocation(line: 16, column: 1,
; CHECK: ![[DEBUG_LOC_ARRAYIDX]] = !DILocation(line: 23, column: 1,
; CHECK: ![[DEBUG_LOC_ARRAYIDX_LOAD]] = !DILocation(line: 24, column: 1,

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "debug_info.pre.ll", directory: "/")
!2 = !{i32 13}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "load_global_from_flat", linkageName: "load_global_from_flat", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !7)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5)
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DILocation(line: 3, column: 1, scope: !5)
!11 = distinct !DISubprogram(name: "simplified_constexpr_gep_addrspacecast", linkageName: "simplified_constexpr_gep_addrspacecast", scope: null, file: !1, line: 4, type: !6, scopeLine: 4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !7)
!12 = !DILocation(line: 4, column: 1, scope: !11)
!13 = !DILocation(line: 5, column: 1, scope: !11)
!14 = !DILocation(line: 6, column: 1, scope: !11)
!15 = !DILocation(line: 7, column: 1, scope: !11)
!16 = distinct !DISubprogram(name: "objectsize_group_to_flat_i32", linkageName: "objectsize_group_to_flat_i32", scope: null, file: !1, line: 8, type: !6, scopeLine: 8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !7)
!17 = !DILocation(line: 8, column: 1, scope: !16)
!18 = !DILocation(line: 9, column: 1, scope: !16)
!19 = !DILocation(line: 10, column: 1, scope: !16)
!20 = distinct !DISubprogram(name: "memset_group_to_flat", linkageName: "memset_group_to_flat", scope: null, file: !1, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !7)
!21 = !DILocation(line: 11, column: 1, scope: !20)
!22 = !DILocation(line: 12, column: 1, scope: !20)
!23 = !{!24, !24, i64 0}
!24 = !{!"A", !25}
!25 = !{!"tbaa root"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"some scope 1"}
!28 = distinct !{!28, !"some domain"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"some scope 2"}
!31 = !DILocation(line: 13, column: 1, scope: !20)
!32 = distinct !DISubprogram(name: "ptrmask_cast_global_to_flat", linkageName: "ptrmask_cast_global_to_flat", scope: null, file: !1, line: 14, type: !6, scopeLine: 14, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !7)
!33 = !DILocation(line: 14, column: 1, scope: !32)
!34 = !DILocation(line: 15, column: 1, scope: !32)
!35 = !DILocation(line: 16, column: 1, scope: !32)
!36 = !DILocation(line: 17, column: 1, scope: !32)
!37 = distinct !DISubprogram(name: "assume_addresspace", linkageName: "assume_addresspace", scope: null, file: !1, line: 18, type: !6, scopeLine: 18, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !7)
!38 = !DILocation(line: 18, column: 1, scope: !37)
!39 = !DILocation(line: 19, column: 1, scope: !37)
!40 = !DILocation(line: 20, column: 1, scope: !37)
!41 = !DILocation(line: 21, column: 1, scope: !37)
!42 = !DILocation(line: 22, column: 1, scope: !37)
!43 = !DILocation(line: 23, column: 1, scope: !37)
!44 = !DILocation(line: 24, column: 1, scope: !37)
!45 = !DILocation(line: 25, column: 1, scope: !37)