; RUN: opt -passes=openmpopt -pass-remarks-analysis=openmp-opt -disable-output < %s 2>&1 | FileCheck %s
; ModuleID = 'declare_target_codegen_globalization.cpp'
source_filename = "declare_target_codegen_globalization.cpp"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.ident_t = type { i32, i32, i32, i32, i8* }
%struct._globalized_locals_ty = type { [32 x i32] }

@0 = private unnamed_addr constant [56 x i8] c";declare_target_codegen_globalization.cpp;maini1;17;1;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 1, i32 0, i8* getelementptr inbounds ([56 x i8], [56 x i8]* @0, i32 0, i32 0) }, align 8
@__omp_offloading_801_3022563__Z6maini1v_l17_exec_mode = weak constant i8 0
@llvm.compiler.used = appending global [1 x i8*] [i8* @__omp_offloading_801_3022563__Z6maini1v_l17_exec_mode], section "llvm.metadata"

; CHECK: remark: declare_target_codegen_globalization.cpp:17:1: Found thread data sharing on the GPU. Expect degraded performance due to data globalization.
; CHECK: remark: declare_target_codegen_globalization.cpp:10:1: Found thread data sharing on the GPU. Expect degraded performance due to data globalization.

; Function Attrs: norecurse nounwind
define weak void @__omp_offloading_801_3022563__Z6maini1v_l17(i32* nonnull align 4 dereferenceable(4) %a) local_unnamed_addr #0 !dbg !10 {
entry:
  %nvptx_num_threads = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x(), !dbg !12, !range !13
  tail call void @__kmpc_spmd_kernel_init(i32 %nvptx_num_threads, i16 1, i16 0) #4, !dbg !12
  tail call void @__kmpc_data_sharing_init_stack_spmd() #4, !dbg !12
  %0 = tail call i32 @__kmpc_global_thread_num(%struct.ident_t* nonnull @1)
  %1 = tail call i8 @__kmpc_is_spmd_exec_mode() #4
  %.not.i.i = icmp eq i8 %1, 0
  br i1 %.not.i.i, label %.non-spmd2.i.i, label %__omp_outlined__.exit

.non-spmd2.i.i:                                   ; preds = %entry
  %2 = tail call i8* @__kmpc_data_sharing_coalesced_push_stack(i64 128, i16 0) #4, !dbg !12
  tail call void @__kmpc_data_sharing_pop_stack(i8* %2) #4, !dbg !14
  br label %__omp_outlined__.exit, !dbg !14

__omp_outlined__.exit:                            ; preds = %entry, %.non-spmd2.i.i
  tail call void @__kmpc_spmd_kernel_deinit_v2(i16 1) #4, !dbg !19
  ret void, !dbg !20
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

declare void @__kmpc_spmd_kernel_init(i32, i16, i16) local_unnamed_addr

declare void @__kmpc_data_sharing_init_stack_spmd() local_unnamed_addr

; Function Attrs: norecurse nounwind readonly
define hidden i32 @_Z3fooRi(i32* nocapture nonnull readonly align 4 dereferenceable(4) %a) local_unnamed_addr #2 !dbg !21 {
entry:
  %0 = load i32, i32* %a, align 4, !dbg !22, !tbaa !23
  ret i32 %0, !dbg !27
}

; Function Attrs: nounwind
define hidden i32 @_Z3barv() local_unnamed_addr #3 !dbg !15 {
entry:
  %a1 = alloca i32, align 4
  %0 = tail call i8 @__kmpc_is_spmd_exec_mode() #4
  %.not = icmp eq i8 %0, 0
  br i1 %.not, label %.non-spmd, label %.exit

.non-spmd:                                        ; preds = %entry
  %1 = tail call i8* @__kmpc_data_sharing_push_stack(i64 128, i16 0) #4, !dbg !31
  %2 = bitcast i8* %1 to %struct._globalized_locals_ty*
  br label %.exit

.exit:                                            ; preds = %entry, %.non-spmd
  %_select_stack = phi %struct._globalized_locals_ty* [ %2, %.non-spmd ], [ null, %entry ]
  %nvptx_tid = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !28
  %nvptx_lane_id = and i32 %nvptx_tid, 31
  %3 = zext i32 %nvptx_lane_id to i64
  %4 = getelementptr inbounds %struct._globalized_locals_ty, %struct._globalized_locals_ty* %_select_stack, i64 0, i32 0, i64 %3
  %5 = select i1 %.not, i32* %4, i32* %a1
  %6 = load i32, i32* %5, align 4, !dbg !29, !tbaa !23
  br i1 %.not, label %.non-spmd2, label %.exit3, !dbg !31

.non-spmd2:                                       ; preds = %.exit
  %7 = bitcast %struct._globalized_locals_ty* %_select_stack to i8*, !dbg !31
  tail call void @__kmpc_data_sharing_pop_stack(i8* %7) #4, !dbg !31
  br label %.exit3, !dbg !31

.exit3:                                           ; preds = %.non-spmd2, %.exit
  ret i32 %6, !dbg !31
}

declare i8 @__kmpc_is_spmd_exec_mode() local_unnamed_addr

declare i8* @__kmpc_data_sharing_coalesced_push_stack(i64, i16) local_unnamed_addr

declare i8* @__kmpc_data_sharing_push_stack(i64, i16) local_unnamed_addr

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

declare void @__kmpc_data_sharing_pop_stack(i8*) local_unnamed_addr

; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(%struct.ident_t*) local_unnamed_addr #4

declare void @__kmpc_spmd_kernel_deinit_v2(i16) local_unnamed_addr

attributes #0 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_35" "target-features"="+ptx32,+sm_35" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { norecurse nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_35" "target-features"="+ptx32,+sm_35" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_35" "target-features"="+ptx32,+sm_35" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!omp_offload.info = !{!3}
!nvvm.annotations = !{!4}
!llvm.module.flags = !{!5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: DebugDirectivesOnly, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "declare_target_codegen_globalization.cpp", directory: "/home/jhuber/Documents/llvm-project/clang/test/OpenMP")
!2 = !{}
!3 = !{i32 0, i32 2049, i32 50472291, !"_Z6maini1v", i32 17, i32 0}
!4 = !{void (i32*)* @__omp_offloading_801_3022563__Z6maini1v_l17, !"kernel", i32 1}
!5 = !{i32 7, !"Dwarf Version", i32 2}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{i32 7, !"PIC Level", i32 2}
!9 = !{!"clang version 12.0.0"}
!10 = distinct !DISubprogram(name: "__omp_offloading_801_3022563__Z6maini1v_l17", scope: !1, file: !1, line: 17, type: !11, scopeLine: 17, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!11 = !DISubroutineType(types: !2)
!12 = !DILocation(line: 17, column: 1, scope: !10)
!13 = !{i32 1, i32 1025}
!14 = !DILocation(line: 10, column: 1, scope: !15, inlinedAt: !16)
!15 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 7, type: !11, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!16 = distinct !DILocation(line: 20, column: 18, scope: !17, inlinedAt: !18)
!17 = distinct !DISubprogram(name: "__omp_outlined__", scope: !1, file: !1, line: 17, type: !11, scopeLine: 17, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!18 = distinct !DILocation(line: 17, column: 1, scope: !10)
!19 = !DILocation(line: 17, column: 40, scope: !10)
!20 = !DILocation(line: 21, column: 3, scope: !10)
!21 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 5, type: !11, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!22 = !DILocation(line: 5, column: 26, scope: !21)
!23 = !{!24, !24, i64 0}
!24 = !{!"int", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C++ TBAA"}
!27 = !DILocation(line: 5, column: 19, scope: !21)
!28 = !{i32 0, i32 1024}
!29 = !DILocation(line: 5, column: 26, scope: !21, inlinedAt: !30)
!30 = distinct !DILocation(line: 9, column: 10, scope: !15)
!31 = !DILocation(line: 10, column: 1, scope: !15)

