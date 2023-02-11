// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-DAG: #[[FOLLOWUP:.*]] = #llvm.loop_annotation<disableNonforced = true>
#followup = #llvm.loop_annotation<disableNonforced = true>

// CHECK-DAG: #[[VECTORIZE:.*]] = #llvm.loop_vectorize<disable = false, predicateEnable = false, scalableEnable = true, width = 16 : i32, followupVectorized = #[[FOLLOWUP]], followupEpilogue = #[[FOLLOWUP]], followupAll = #[[FOLLOWUP]]>
#vectorize = #llvm.loop_vectorize<
  disable = false, predicateEnable = false, scalableEnable = true, width = 16 : i32,
  followupVectorized = #followup, followupEpilogue = #followup, followupAll = #followup
>

// CHECK-DAG: #[[INTERLEAVE:.*]] = #llvm.loop_interleave<count = 32 : i32>
#interleave = #llvm.loop_interleave<count = 32 : i32>

// CHECK-DAG: #[[UNROLL:.*]] = #llvm.loop_unroll<disable = true, count = 32 : i32, runtimeDisable = true, full = false, followupUnrolled = #[[FOLLOWUP]], followupRemainder = #[[FOLLOWUP]], followupAll = #[[FOLLOWUP]]>
#unroll = #llvm.loop_unroll<
  disable = true, count = 32 : i32, runtimeDisable = true, full = false,
  followupUnrolled = #followup, followupRemainder = #followup, followupAll = #followup
>

// CHECK-DAG: #[[UNROLL_AND_JAM:.*]] = #llvm.loop_unroll_and_jam<disable = false, count = 16 : i32, followupOuter = #[[FOLLOWUP]], followupInner = #[[FOLLOWUP]], followupRemainderOuter = #[[FOLLOWUP]], followupRemainderInner = #[[FOLLOWUP]], followupAll = #[[FOLLOWUP]]>
#unrollAndJam = #llvm.loop_unroll_and_jam<
  disable = false, count = 16 : i32, followupOuter = #followup, followupInner = #followup,
  followupRemainderOuter = #followup, followupRemainderInner = #followup, followupAll = #followup
>

// CHECK-DAG: #[[LICM:.*]] = #llvm.loop_licm<disable = false, versioningDisable = true>
#licm = #llvm.loop_licm<disable = false, versioningDisable = true>

// CHECK-DAG: #[[DISTRIBUTE:.*]] = #llvm.loop_distribute<disable = true, followupCoincident = #[[FOLLOWUP]], followupSequential = #[[FOLLOWUP]], followupFallback = #[[FOLLOWUP]], followupAll = #[[FOLLOWUP]]>
#distribute = #llvm.loop_distribute<
  disable = true, followupCoincident = #followup, followupSequential = #followup,
  followupFallback = #followup, followupAll = #followup
>

// CHECK-DAG: #[[PIPELINE:.*]] = #llvm.loop_pipeline<disable = true, initiationinterval = 1 : i32>
#pipeline = #llvm.loop_pipeline<disable = true, initiationinterval = 1 : i32>

// CHECK: #[[LOOP_ANNOT:.*]] = #llvm.loop_annotation<
// CHECK-DAG: disableNonforced = false
// CHECK-DAG: mustProgress = true
// CHECK-DAG: unroll = #[[UNROLL]]
// CHECK-DAG: unrollAndJam = #[[UNROLL_AND_JAM]]
// CHECK-DAG: licm = #[[LICM]]
// CHECK-DAG: distribute = #[[DISTRIBUTE]]
// CHECK-DAG: pipeline = #[[PIPELINE]]
// CHECK-DAG: isVectorized = false
// CHECK-DAG: parallelAccesses = @metadata::@group1, @metadata::@group2>
#loopMD = #llvm.loop_annotation<disableNonforced = false,
        mustProgress = true,
        vectorize = #vectorize,
        interleave = #interleave,
        unroll = #unroll,
        unrollAndJam = #unrollAndJam,
        licm = #licm,
        distribute = #distribute,
        pipeline = #pipeline,
        isVectorized = false,
        parallelAccesses = @metadata::@group1, @metadata::@group2>

// CHECK: llvm.func @loop_annotation
llvm.func @loop_annotation() {
  // CHECK: llvm.br ^bb1 {llvm.loop = #[[LOOP_ANNOT]]
  llvm.br ^bb1 {llvm.loop = #loopMD}
^bb1:
  llvm.return
}

llvm.metadata @metadata {
  llvm.access_group @group1
  llvm.access_group @group2
}
