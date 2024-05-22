; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-DAG: #[[$GROUP0:.*]] = #llvm.access_group<id = {{.*}}>
; CHECK-DAG: #[[$GROUP1:.*]] = #llvm.access_group<id = {{.*}}>
; CHECK-DAG: #[[$GROUP2:.*]] = #llvm.access_group<id = {{.*}}>
; CHECK-DAG: #[[$GROUP3:.*]] = #llvm.access_group<id = {{.*}}>

; CHECK-LABEL: llvm.func @access_group
define void @access_group(ptr %arg1) {
  ; CHECK:  access_groups = [#[[$GROUP0]], #[[$GROUP1]]]
  %1 = load i32, ptr %arg1, !llvm.access.group !0
  ; CHECK:  access_groups = [#[[$GROUP2]], #[[$GROUP0]]]
  %2 = load i32, ptr %arg1, !llvm.access.group !1
  ; CHECK:  access_groups = [#[[$GROUP3]]]
  %3 = load i32, ptr %arg1, !llvm.access.group !2
  ret void
}

!0 = !{!3, !4}
!1 = !{!5, !3}
!2 = distinct !{}
!3 = distinct !{}
!4 = distinct !{}
!5 = distinct !{}

; // -----

; CHECK-LABEL: llvm.func @supported_ops
define void @supported_ops(ptr %arg1, float %arg2, i32 %arg3, i32 %arg4) {
  ; CHECK: llvm.load {{.*}}access_groups =
  %1 = load i32, ptr %arg1, !llvm.access.group !0
  ; CHECK: llvm.store {{.*}}access_groups =
  store i32 %1, ptr %arg1, !llvm.access.group !0
  ; CHECK: llvm.atomicrmw {{.*}}access_groups =
  %2 = atomicrmw fmax ptr %arg1, float %arg2 acquire, !llvm.access.group !0
  ; CHECK: llvm.cmpxchg {{.*}}access_groups =
  %3 = cmpxchg ptr %arg1, i32 %arg3, i32 %arg4 monotonic seq_cst, !llvm.access.group !0
  ; CHECK: "llvm.intr.memcpy"{{.*}}access_groups =
  call void @llvm.memcpy.p0.p0.i32(ptr %arg1, ptr %arg1, i32 4, i1 false), !llvm.access.group !0
  ; CHECK: "llvm.intr.memset"{{.*}}access_groups =
  call void @llvm.memset.p0.i32(ptr %arg1, i8 42, i32 4, i1 false), !llvm.access.group !0
  ; CHECK: llvm.call{{.*}}access_groups =
  call void @foo(ptr %arg1), !llvm.access.group !0
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1 immarg)
declare void @foo(ptr %arg1)

!0 = !{!1, !2}
!1 = distinct !{}
!2 = distinct !{}

; // -----

; CHECK: #[[$ANNOT_ATTR:.*]] = #llvm.loop_annotation<disableNonforced = true, mustProgress = true, isVectorized = true>

; CHECK-LABEL: @simple
define void @simple(i64 %n, ptr %A) {
entry:
; CHECK: llvm.br ^{{.*}} {loop_annotation = #[[$ANNOT_ATTR]]}
  br label %end, !llvm.loop !1
end:
  ret void
}

!1 = distinct !{!1, !2, !3, !4}
!2 = !{!"llvm.loop.disable_nonforced"}
!3 = !{!"llvm.loop.mustprogress"}
!4 = !{!"llvm.loop.isvectorized", i32 1}

; // -----

; CHECK-DAG: #[[FOLLOWUP:.*]] = #llvm.loop_annotation<disableNonforced = true>
; CHECK-DAG: #[[VECTORIZE_ATTR:.*]] = #llvm.loop_vectorize<disable = false, predicateEnable = true, scalableEnable = false, width = 16 : i32, followupVectorized = #[[FOLLOWUP]], followupEpilogue = #[[FOLLOWUP]], followupAll = #[[FOLLOWUP]]>
; CHECK-DAG: #[[$ANNOT_ATTR:.*]] = #llvm.loop_annotation<vectorize = #[[VECTORIZE_ATTR]]>

; CHECK-LABEL: @vectorize
define void @vectorize(i64 %n, ptr %A) {
entry:
; CHECK: llvm.br ^{{.*}} {loop_annotation = #[[$ANNOT_ATTR]]}
  br label %end, !llvm.loop !1
end:
  ret void
}

!1 = distinct !{!1, !2, !3, !4, !5, !6, !7, !8}
!2 = !{!"llvm.loop.vectorize.enable", i1 1}
!3 = !{!"llvm.loop.vectorize.predicate.enable", i1 1}
!4 = !{!"llvm.loop.vectorize.scalable.enable", i1 0}
!5 = !{!"llvm.loop.vectorize.width", i32 16}
!6 = !{!"llvm.loop.vectorize.followup_vectorized", !9}
!7 = !{!"llvm.loop.vectorize.followup_epilogue", !9}
!8 = !{!"llvm.loop.vectorize.followup_all", !9}

!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.disable_nonforced"}

; // -----

; CHECK-DAG: #[[INTERLEAVE_ATTR:.*]] = #llvm.loop_interleave<count = 8 : i32>
; CHECK-DAG: #[[$ANNOT_ATTR:.*]] = #llvm.loop_annotation<interleave = #[[INTERLEAVE_ATTR]]>

; CHECK-LABEL: @interleave
define void @interleave(i64 %n, ptr %A) {
entry:
; CHECK: llvm.br ^{{.*}} {loop_annotation = #[[$ANNOT_ATTR]]}
  br label %end, !llvm.loop !1
end:
  ret void
}

!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.interleave.count", i32 8}

; // -----

; CHECK-DAG: #[[FOLLOWUP:.*]] = #llvm.loop_annotation<disableNonforced = true>
; CHECK-DAG: #[[UNROLL_ATTR:.*]] = #llvm.loop_unroll<disable = false, count = 16 : i32, runtimeDisable = true, full = true, followupUnrolled = #[[FOLLOWUP]], followupRemainder = #[[FOLLOWUP]], followupAll = #[[FOLLOWUP]]>
; CHECK-DAG: #[[$ANNOT_ATTR:.*]] = #llvm.loop_annotation<unroll = #[[UNROLL_ATTR]]>

; CHECK-LABEL: @unroll
define void @unroll(i64 %n, ptr %A) {
entry:
; CHECK: llvm.br ^{{.*}} {loop_annotation = #[[$ANNOT_ATTR]]}
  br label %end, !llvm.loop !1
end:
  ret void
}

!1 = distinct !{!1, !2, !3, !4, !5, !6, !7, !8}
!2 = !{!"llvm.loop.unroll.enable"}
!3 = !{!"llvm.loop.unroll.count", i32 16}
!4 = !{!"llvm.loop.unroll.runtime.disable"}
!5 = !{!"llvm.loop.unroll.full"}
!6 = !{!"llvm.loop.unroll.followup_unrolled", !9}
!7 = !{!"llvm.loop.unroll.followup_remainder", !9}
!8 = !{!"llvm.loop.unroll.followup_all", !9}

!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.disable_nonforced"}

; // -----

; CHECK-DAG: #[[UNROLL_ATTR:.*]] = #llvm.loop_unroll<disable = true>
; CHECK-DAG: #[[$ANNOT_ATTR:.*]] = #llvm.loop_annotation<unroll = #[[UNROLL_ATTR]]>

; CHECK-LABEL: @unroll_disable
define void @unroll_disable(i64 %n, ptr %A) {
entry:
; CHECK: llvm.br ^{{.*}} {loop_annotation = #[[$ANNOT_ATTR]]}
  br label %end, !llvm.loop !1
end:
  ret void
}

!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.unroll.disable"}

; // -----

; CHECK-DAG: #[[FOLLOWUP:.*]] = #llvm.loop_annotation<disableNonforced = true>
; CHECK-DAG: #[[UNROLL_AND_JAM_ATTR:.*]] = #llvm.loop_unroll_and_jam<disable = false, count = 32 : i32, followupOuter = #[[FOLLOWUP]], followupInner = #[[FOLLOWUP]], followupRemainderOuter = #[[FOLLOWUP]], followupRemainderInner = #[[FOLLOWUP]], followupAll = #[[FOLLOWUP]]>
; CHECK-DAG: #[[$ANNOT_ATTR:.*]] = #llvm.loop_annotation<unrollAndJam = #[[UNROLL_AND_JAM_ATTR]]>

; CHECK-LABEL: @unroll_and_jam
define void @unroll_and_jam(i64 %n, ptr %A) {
entry:
; CHECK: llvm.br ^{{.*}} {loop_annotation = #[[$ANNOT_ATTR]]}
  br label %end, !llvm.loop !1
end:
  ret void
}

!1 = distinct !{!1, !2, !3, !4, !5, !6, !7, !8}
!2 = !{!"llvm.loop.unroll_and_jam.enable"}
!3 = !{!"llvm.loop.unroll_and_jam.count", i32 32}
!4 = !{!"llvm.loop.unroll_and_jam.followup_outer", !9}
!5 = !{!"llvm.loop.unroll_and_jam.followup_inner", !9}
!6 = !{!"llvm.loop.unroll_and_jam.followup_remainder_outer", !9}
!7 = !{!"llvm.loop.unroll_and_jam.followup_remainder_inner", !9}
!8 = !{!"llvm.loop.unroll_and_jam.followup_all", !9}

!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.disable_nonforced"}

; // -----

; CHECK-DAG: #[[LICM_ATTR:.*]] = #llvm.loop_licm<disable = true, versioningDisable = true>
; CHECK-DAG: #[[$ANNOT_ATTR:.*]] = #llvm.loop_annotation<licm = #[[LICM_ATTR]]>

; CHECK-LABEL: @licm
define void @licm(i64 %n, ptr %A) {
entry:
; CHECK: llvm.br ^{{.*}} {loop_annotation = #[[$ANNOT_ATTR]]}
  br label %end, !llvm.loop !1
end:
  ret void
}

!1 = distinct !{!1, !2, !3}
!2 = !{!"llvm.licm.disable"}
!3 = !{!"llvm.loop.licm_versioning.disable"}

; // -----

; CHECK-DAG: #[[FOLLOWUP:.*]] = #llvm.loop_annotation<disableNonforced = true>
; CHECK-DAG: #[[DISTRIBUTE_ATTR:.*]] = #llvm.loop_distribute<disable = true, followupCoincident = #[[FOLLOWUP]], followupSequential = #[[FOLLOWUP]], followupFallback = #[[FOLLOWUP]], followupAll = #[[FOLLOWUP]]>
; CHECK-DAG: #[[$ANNOT_ATTR:.*]] = #llvm.loop_annotation<distribute = #[[DISTRIBUTE_ATTR]]>

; CHECK-LABEL: @distribute
define void @distribute(i64 %n, ptr %A) {
entry:
; CHECK: llvm.br ^{{.*}} {loop_annotation = #[[$ANNOT_ATTR]]}
  br label %end, !llvm.loop !1
end:
  ret void
}

!1 = distinct !{!1, !2, !3, !4, !5, !6}
!2 = !{!"llvm.loop.distribute.enable", i1 0}
!3 = !{!"llvm.loop.distribute.followup_coincident", !9}
!4 = !{!"llvm.loop.distribute.followup_sequential", !9}
!5 = !{!"llvm.loop.distribute.followup_fallback", !9}
!6 = !{!"llvm.loop.distribute.followup_all", !9}

!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.disable_nonforced"}

; // -----

; CHECK-DAG: #[[PIPELINE_ATTR:.*]] = #llvm.loop_pipeline<disable = false, initiationinterval = 2 : i32>
; CHECK-DAG: #[[$ANNOT_ATTR:.*]] = #llvm.loop_annotation<pipeline = #[[PIPELINE_ATTR]]>

; CHECK-LABEL: @pipeline
define void @pipeline(i64 %n, ptr %A) {
entry:
; CHECK: llvm.br ^{{.*}} {loop_annotation = #[[$ANNOT_ATTR]]}
  br label %end, !llvm.loop !1
end:
  ret void
}

!1 = distinct !{!1, !2, !3}
!2 = !{!"llvm.loop.pipeline.disable", i1 0}
!3 = !{!"llvm.loop.pipeline.initiationinterval", i32 2}

; // -----

; CHECK-DAG: #[[PEELED_ATTR:.*]] = #llvm.loop_peeled<count = 5 : i32>
; CHECK-DAG: #[[$ANNOT_ATTR:.*]] = #llvm.loop_annotation<peeled = #[[PEELED_ATTR]]>

; CHECK-LABEL: @peeled
define void @peeled(i64 %n, ptr %A) {
entry:
; CHECK: llvm.br ^{{.*}} {loop_annotation = #[[$ANNOT_ATTR]]}
  br label %end, !llvm.loop !1
end:
  ret void
}

!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.peeled.count", i32 5}

; // -----

; CHECK-DAG: #[[UNSWITCH_ATTR:.*]] = #llvm.loop_unswitch<partialDisable = true>
; CHECK-DAG: #[[$ANNOT_ATTR:.*]] = #llvm.loop_annotation<unswitch = #[[UNSWITCH_ATTR]]>

; CHECK-LABEL: @unswitched
define void @unswitched(i64 %n, ptr %A) {
entry:
; CHECK: llvm.br ^{{.*}} {loop_annotation = #[[$ANNOT_ATTR]]}
  br label %end, !llvm.loop !1
end:
  ret void
}

!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.unswitch.partial.disable"}

; // -----

; CHECK: #[[GROUP0:.*]] = #llvm.access_group<id = {{.*}}>
; CHECK: #[[$ANNOT_ATTR:.*]] = #llvm.loop_annotation<parallelAccesses = #[[GROUP0]]>

; CHECK-LABEL: @parallel_accesses
define void @parallel_accesses(ptr %arg) {
entry:
  %0 = load i32, ptr %arg, !llvm.access.group !0
; CHECK: llvm.br ^{{.*}} {loop_annotation = #[[$ANNOT_ATTR]]}
  br label %end, !llvm.loop !1
end:
  ret void
}

!0 = distinct !{}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.parallel_accesses", !0}

; // -----

; CHECK: #[[GROUP0:.*]] = #llvm.access_group<id = {{.*}}>
; CHECK: #[[GROUP1:.*]] = #llvm.access_group<id = {{.*}}>
; CHECK: #[[$ANNOT_ATTR:.*]] = #llvm.loop_annotation<parallelAccesses = #[[GROUP0]], #[[GROUP1]]>

; CHECK-LABEL: @multiple_parallel_accesses
define void @multiple_parallel_accesses(ptr %arg) {
entry:
  %0 = load i32, ptr %arg, !llvm.access.group !0
  %1 = load i32, ptr %arg, !llvm.access.group !3
; CHECK: llvm.br ^{{.*}} {loop_annotation = #[[$ANNOT_ATTR]]}
  br label %end, !llvm.loop !1
end:
  ret void
}

!0 = distinct !{}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.parallel_accesses", !0, !3}
!3 = distinct !{}

; // -----

; Verify the unused access group is not imported.
; CHECK-COUNT1: #llvm.access_group

; CHECK-LABEL: @unused_parallel_access
define void @unused_parallel_access(ptr %arg) {
entry:
  %0 = load i32, ptr %arg, !llvm.access.group !0
  br label %end, !llvm.loop !1
end:
  ret void
}

!0 = distinct !{}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.parallel_accesses", !0, !3}
!3 = distinct !{}

; // -----

; CHECK: #[[start_loc:.*]] = loc("metadata-loop.ll":1:2)
; CHECK: #[[end_loc:.*]] = loc("metadata-loop.ll":2:2)
; CHECK: #[[SUBPROGRAM:.*]] = #llvm.di_subprogram<
; CHECK: #[[start_loc_fused:.*]] = loc(fused<#[[SUBPROGRAM]]>[#[[start_loc]]])
; CHECK: #[[end_loc_fused:.*]] = loc(fused<#[[SUBPROGRAM]]>[#[[end_loc]]])
; CHECK: #[[$ANNOT_ATTR:.*]] = #llvm.loop_annotation<
; CHECK-SAME: mustProgress = true
; CHECK-SAME: startLoc = #[[start_loc_fused]]
; CHECK-SAME: endLoc = #[[end_loc_fused]]

; CHECK-LABEL: @loop_locs
define void @loop_locs(i64 %n, ptr %A) {
entry:
; CHECK: llvm.br ^{{.*}} {loop_annotation = #[[$ANNOT_ATTR]]}
  br label %end, !llvm.loop !6
end:
  ret void
}

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2)
!2 = !DIFile(filename: "metadata-loop.ll", directory: "/")
!3 = distinct !DISubprogram(name: "loop_locs", scope: !2, file: !2, spFlags: DISPFlagDefinition, unit: !1)
!4 = !DILocation(line: 1, column: 2, scope: !3)
!5 = !DILocation(line: 2, column: 2, scope: !3)

!6 = distinct !{!6, !4, !5, !7}
!7 = !{!"llvm.loop.mustprogress"}
