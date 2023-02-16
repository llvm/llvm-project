; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK: llvm.metadata @__llvm_global_metadata {
; CHECK:   llvm.access_group @[[$GROUP0:.*]]
; CHECK:   llvm.access_group @[[$GROUP1:.*]]
; CHECK:   llvm.access_group @[[$GROUP2:.*]]
; CHECK:   llvm.access_group @[[$GROUP3:.*]]
; CHECK: }

; CHECK-LABEL: llvm.func @access_group
define void @access_group(ptr %arg1) {
  ; CHECK:  access_groups = [@__llvm_global_metadata::@[[$GROUP0]], @__llvm_global_metadata::@[[$GROUP1]]]
  %1 = load i32, ptr %arg1, !llvm.access.group !0
  ; CHECK:  access_groups = [@__llvm_global_metadata::@[[$GROUP2]], @__llvm_global_metadata::@[[$GROUP0]]]
  %2 = load i32, ptr %arg1, !llvm.access.group !1
  ; CHECK:  access_groups = [@__llvm_global_metadata::@[[$GROUP3]]]
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

; CHECK: #[[$ANNOT_ATTR:.*]] = #llvm.loop_annotation<disableNonforced = true, mustProgress = true, isVectorized = true>

; CHECK-LABEL: @simple
define void @simple(i64 %n, ptr %A) {
entry:
; CHECK: llvm.br ^{{.*}} {llvm.loop = #[[$ANNOT_ATTR]]}
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
; CHECK: llvm.br ^{{.*}} {llvm.loop = #[[$ANNOT_ATTR]]}
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
; CHECK: llvm.br ^{{.*}} {llvm.loop = #[[$ANNOT_ATTR]]}
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
; CHECK: llvm.br ^{{.*}} {llvm.loop = #[[$ANNOT_ATTR]]}
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
; CHECK: llvm.br ^{{.*}} {llvm.loop = #[[$ANNOT_ATTR]]}
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
; CHECK: llvm.br ^{{.*}} {llvm.loop = #[[$ANNOT_ATTR]]}
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
; CHECK: llvm.br ^{{.*}} {llvm.loop = #[[$ANNOT_ATTR]]}
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
; CHECK: llvm.br ^{{.*}} {llvm.loop = #[[$ANNOT_ATTR]]}
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
; CHECK: llvm.br ^{{.*}} {llvm.loop = #[[$ANNOT_ATTR]]}
  br label %end, !llvm.loop !1
end:
  ret void
}

!1 = distinct !{!1, !2, !3}
!2 = !{!"llvm.loop.pipeline.disable", i1 0}
!3 = !{!"llvm.loop.pipeline.initiationinterval", i32 2}

; // -----

; CHECK: #[[$ANNOT_ATTR:.*]] = #llvm.loop_annotation<parallelAccesses = @__llvm_global_metadata::@[[GROUP0:.*]]>

; CHECK: llvm.metadata @__llvm_global_metadata {
; CHECK:   llvm.access_group @[[GROUP0]]

; CHECK-LABEL: @parallel_accesses
define void @parallel_accesses(ptr %arg) {
entry:
  %0 = load i32, ptr %arg, !llvm.access.group !0
; CHECK: llvm.br ^{{.*}} {llvm.loop = #[[$ANNOT_ATTR]]}
  br label %end, !llvm.loop !1
end:
  ret void
}

!0 = distinct !{}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.parallel_accesses", !0}

; // -----

; CHECK: #[[$ANNOT_ATTR:.*]] = #llvm.loop_annotation<parallelAccesses = @__llvm_global_metadata::@[[GROUP0:.*]], @__llvm_global_metadata::@[[GROUP1:.*]]>

; CHECK: llvm.metadata @__llvm_global_metadata {
; CHECK:   llvm.access_group @[[GROUP0]]
; CHECK:   llvm.access_group @[[GROUP1]]

; CHECK-LABEL: @multiple_parallel_accesses
define void @multiple_parallel_accesses(ptr %arg) {
entry:
  %0 = load i32, ptr %arg, !llvm.access.group !0
  %1 = load i32, ptr %arg, !llvm.access.group !3
; CHECK: llvm.br ^{{.*}} {llvm.loop = #[[$ANNOT_ATTR]]}
  br label %end, !llvm.loop !1
end:
  ret void
}

!0 = distinct !{}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.parallel_accesses", !0, !3}
!3 = distinct !{}
