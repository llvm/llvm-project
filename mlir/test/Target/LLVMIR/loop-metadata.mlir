// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK-LABEL: @disableNonForced
llvm.func @disableNonForced() {
  // CHECK: br {{.*}} !llvm.loop ![[LOOP_NODE:[0-9]+]]
  llvm.br ^bb1 {llvm.loop = #llvm.loop_annotation<disableNonforced = true>}
^bb1:
  llvm.return
}

// CHECK: ![[LOOP_NODE]] = distinct !{![[LOOP_NODE]], !{{[0-9]+}}}
// CHECK-DAG: ![[VEC_NODE0:[0-9]+]] = !{!"llvm.loop.disable_nonforced"}

// -----

// CHECK-LABEL: @mustprogress
llvm.func @mustprogress() {
  // CHECK: br {{.*}} !llvm.loop ![[LOOP_NODE:[0-9]+]]
  llvm.br ^bb1 {llvm.loop = #llvm.loop_annotation<mustProgress = true>}
^bb1:
  llvm.return
}

// CHECK: ![[LOOP_NODE]] = distinct !{![[LOOP_NODE]], !{{[0-9]+}}}
// CHECK-DAG: ![[VEC_NODE0:[0-9]+]] = !{!"llvm.loop.mustprogress"}

// -----

// CHECK-LABEL: @isvectorized
llvm.func @isvectorized() {
  // CHECK: br {{.*}} !llvm.loop ![[LOOP_NODE:[0-9]+]]
  llvm.br ^bb1 {llvm.loop = #llvm.loop_annotation<isVectorized = true>}
^bb1:
  llvm.return
}

// CHECK: ![[LOOP_NODE]] = distinct !{![[LOOP_NODE]], !{{[0-9]+}}}
// CHECK-DAG: ![[VEC_NODE0:[0-9]+]] = !{!"llvm.loop.isvectorized", i32 1}

// -----

#followup = #llvm.loop_annotation<disableNonforced = true>

// CHECK-LABEL: @vectorizeOptions
llvm.func @vectorizeOptions() {
  // CHECK: br {{.*}} !llvm.loop ![[LOOP_NODE:[0-9]+]]
  llvm.br ^bb1 {llvm.loop = #llvm.loop_annotation<vectorize = <
    disable = false, predicateEnable = true, scalableEnable = false, width = 16 : i32, 
    followupVectorized = #followup, followupEpilogue = #followup, followupAll = #followup>
  >}
^bb1:
  llvm.return
}

// CHECK-DAG: ![[NON_FORCED:[0-9]+]] = !{!"llvm.loop.disable_nonforced"}
// CHECK-DAG: ![[FOLLOWUP:[0-9]+]] = distinct !{![[FOLLOWUP]], ![[NON_FORCED]]}
// CHECK-DAG: ![[LOOP_NODE]] = distinct !{![[LOOP_NODE]], !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.vectorize.enable", i1 true}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.vectorize.predicate.enable", i1 true}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.vectorize.scalable.enable", i1 false}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.vectorize.width", i32 16}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.vectorize.followup_vectorized", ![[FOLLOWUP]]}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.vectorize.followup_epilogue", ![[FOLLOWUP]]}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.vectorize.followup_all", ![[FOLLOWUP]]}

// -----

// CHECK-LABEL: @interleaveOptions
llvm.func @interleaveOptions() {
  // CHECK: br {{.*}} !llvm.loop ![[LOOP_NODE:[0-9]+]]
  llvm.br ^bb1 {llvm.loop = #llvm.loop_annotation<interleave = <count = 32 : i32>>}
^bb1:
  llvm.return
}

// CHECK: ![[LOOP_NODE]] = distinct !{![[LOOP_NODE]], ![[INTERLEAVE_NODE:[0-9]+]]}
// CHECK: ![[INTERLEAVE_NODE]] = !{!"llvm.loop.interleave.count", i32 32}

// -----

#followup = #llvm.loop_annotation<disableNonforced = true>

// CHECK-LABEL: @unrollOptions
llvm.func @unrollOptions() {
  // CHECK: br {{.*}} !llvm.loop ![[LOOP_NODE:[0-9]+]]
  llvm.br ^bb1 {llvm.loop = #llvm.loop_annotation<unroll = <
    disable = true, count = 64 : i32, runtimeDisable = false, full = false,
    followupUnrolled = #followup, followupRemainder = #followup, followupAll = #followup>
  >}
^bb1:
  llvm.return
}

// CHECK-DAG: ![[NON_FORCED:[0-9]+]] = !{!"llvm.loop.disable_nonforced"}
// CHECK-DAG: ![[FOLLOWUP:[0-9]+]] = distinct !{![[FOLLOWUP]], ![[NON_FORCED]]}
// CHECK-DAG: ![[LOOP_NODE]] = distinct !{![[LOOP_NODE]], !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.unroll.disable"}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.unroll.count", i32 64}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.unroll.runtime.disable", i1 false}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.unroll.followup_unrolled", ![[FOLLOWUP]]}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.unroll.followup_remainder", ![[FOLLOWUP]]}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.unroll.followup_all", ![[FOLLOWUP]]}

// -----

// CHECK-LABEL: @unrollOptions2
llvm.func @unrollOptions2() {
  // CHECK: br {{.*}} !llvm.loop ![[LOOP_NODE:[0-9]+]]
  llvm.br ^bb1 {llvm.loop = #llvm.loop_annotation<unroll = <disable = false, full = true>>}
^bb1:
  llvm.return
}

// CHECK: ![[LOOP_NODE]] = distinct !{![[LOOP_NODE]], !{{[0-9]+}}, !{{[0-9]+}}} 
// CHECK-DAG: ![[VEC_NODE0:[0-9]+]] = !{!"llvm.loop.unroll.enable"}
// CHECK-DAG: ![[VEC_NODE2:[0-9]+]] = !{!"llvm.loop.unroll.full"}

// -----

#followup = #llvm.loop_annotation<disableNonforced = true>

// CHECK-LABEL: @unrollAndJamOptions
llvm.func @unrollAndJamOptions() {
  // CHECK: br {{.*}} !llvm.loop ![[LOOP_NODE:[0-9]+]]
  llvm.br ^bb1 {llvm.loop = #llvm.loop_annotation<unrollAndJam = <
    disable = false, count = 8 : i32, followupOuter = #followup, followupInner = #followup,
    followupRemainderOuter = #followup, followupRemainderInner = #followup, followupAll = #followup>
  >}
^bb1:
  llvm.return
}

// CHECK-DAG: ![[NON_FORCED:[0-9]+]] = !{!"llvm.loop.disable_nonforced"}
// CHECK-DAG: ![[FOLLOWUP:[0-9]+]] = distinct !{![[FOLLOWUP]], ![[NON_FORCED]]}
// CHECK-DAG: ![[LOOP_NODE]] = distinct !{![[LOOP_NODE]], !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.unroll_and_jam.enable"}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.unroll_and_jam.count", i32 8}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.unroll_and_jam.followup_outer", ![[FOLLOWUP]]}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.unroll_and_jam.followup_inner", ![[FOLLOWUP]]}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.unroll_and_jam.followup_remainder_outer", ![[FOLLOWUP]]}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.unroll_and_jam.followup_remainder_inner", ![[FOLLOWUP]]}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.unroll_and_jam.followup_all", ![[FOLLOWUP]]}

// -----

// CHECK-LABEL: @licmOptions
llvm.func @licmOptions() {
  // CHECK: br {{.*}} !llvm.loop ![[LOOP_NODE:[0-9]+]]
  llvm.br ^bb1 {llvm.loop = #llvm.loop_annotation<licm = <disable = false, versioningDisable = true>>}
^bb1:
  llvm.return
}

// CHECK: ![[LOOP_NODE]] = distinct !{![[LOOP_NODE]], !{{[0-9]+}}}
// CHECK-DAG: ![[VEC_NODE0:[0-9]+]] = !{!"llvm.loop.licm_versioning.disable"}

// -----

// CHECK-LABEL: @licmOptions2
llvm.func @licmOptions2() {
  // CHECK: br {{.*}} !llvm.loop ![[LOOP_NODE:[0-9]+]]
  llvm.br ^bb1 {llvm.loop = #llvm.loop_annotation<licm = <disable = true, versioningDisable = false>>}
^bb1:
  llvm.return
}

// CHECK: ![[LOOP_NODE]] = distinct !{![[LOOP_NODE]], !{{[0-9]+}}}
// CHECK-DAG: ![[VEC_NODE0:[0-9]+]] = !{!"llvm.licm.disable"}

// -----

#followup = #llvm.loop_annotation<disableNonforced = true>

// CHECK-LABEL: @distributeOptions
llvm.func @distributeOptions() {
  // CHECK: br {{.*}} !llvm.loop ![[LOOP_NODE:[0-9]+]]
  llvm.br ^bb1 {llvm.loop = #llvm.loop_annotation<distribute = <
    disable = true, followupCoincident = #followup, followupSequential = #followup,
    followupFallback = #followup, followupAll = #followup>
  >}
^bb1:
  llvm.return
}

// CHECK-DAG: ![[NON_FORCED:[0-9]+]] = !{!"llvm.loop.disable_nonforced"}
// CHECK-DAG: ![[FOLLOWUP:[0-9]+]] = distinct !{![[FOLLOWUP]], ![[NON_FORCED]]}
// CHECK-DAG: ![[LOOP_NODE]] = distinct !{![[LOOP_NODE]], !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.distribute.enable", i1 false}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.distribute.followup_coincident", ![[FOLLOWUP]]}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.distribute.followup_sequential", ![[FOLLOWUP]]}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.distribute.followup_fallback", ![[FOLLOWUP]]}
// CHECK-DAG: !{{[0-9]+}} = !{!"llvm.loop.distribute.followup_all", ![[FOLLOWUP]]}

// -----

// CHECK-LABEL: @pipelineOptions
llvm.func @pipelineOptions() {
  // CHECK: br {{.*}} !llvm.loop ![[LOOP_NODE:[0-9]+]]
  llvm.br ^bb1 {llvm.loop = #llvm.loop_annotation<pipeline = <disable = false, initiationinterval = 1 : i32>>}
^bb1:
  llvm.return
}

// CHECK: ![[LOOP_NODE]] = distinct !{![[LOOP_NODE]], !{{[0-9]+}}, !{{[0-9]+}}}
// CHECK-DAG: ![[VEC_NODE0:[0-9]+]] = !{!"llvm.loop.pipeline.disable", i1 false}
// CHECK-DAG: ![[VEC_NODE0:[0-9]+]] = !{!"llvm.loop.pipeline.initiationinterval", i32 1}

// -----

// CHECK-LABEL: @loopOptions
llvm.func @loopOptions(%arg1 : i32, %arg2 : i32) {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.alloca %arg1 x i32 : (i32) -> (!llvm.ptr<i32>)
    llvm.br ^bb3(%0 : i32)
  ^bb3(%1: i32):
    %2 = llvm.icmp "slt" %1, %arg1 : i32
    // CHECK: br i1 {{.*}} !llvm.loop ![[LOOP_NODE:[0-9]+]]
    llvm.cond_br %2, ^bb4, ^bb5 {llvm.loop = #llvm.loop_annotation<
          licm = <disable = true>,
          interleave = <count = 1>,
          unroll = <disable = true>, pipeline = <disable = true, initiationinterval = 2>,
          parallelAccesses = @metadata::@group1, @metadata::@group2>}
  ^bb4:
    %3 = llvm.add %1, %arg2  : i32
    // CHECK: = load i32, ptr %{{.*}} !llvm.access.group ![[ACCESS_GROUPS_NODE:[0-9]+]]
    %5 = llvm.load %4 { access_groups = [@metadata::@group1, @metadata::@group2] } : !llvm.ptr<i32>
    // CHECK: br label {{.*}} !llvm.loop ![[LOOP_NODE]]
    llvm.br ^bb3(%3 : i32) {llvm.loop = #llvm.loop_annotation<
          licm = <disable = true>,
          interleave = <count = 1>,
          unroll = <disable = true>, pipeline = <disable = true, initiationinterval = 2>,
          parallelAccesses = @metadata::@group1, @metadata::@group2>}

  ^bb5:
    llvm.return
}

llvm.metadata @metadata {
  llvm.access_group @group1
  llvm.access_group @group2
}

// CHECK: ![[LOOP_NODE]] = distinct !{![[LOOP_NODE]], !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}}       
// CHECK-DAG: ![[PA_NODE:[0-9]+]] = !{!"llvm.loop.parallel_accesses", ![[GROUP_NODE1:[0-9]+]], ![[GROUP_NODE2:[0-9]+]]}
// CHECK-DAG: ![[GROUP_NODE1:[0-9]+]] = distinct !{}
// CHECK-DAG: ![[GROUP_NODE2:[0-9]+]] = distinct !{}
// CHECK-DAG: ![[UNROLL_DISABLE_NODE:[0-9]+]] = !{!"llvm.loop.unroll.disable"}
// CHECK-DAG: ![[LICM_DISABLE_NODE:[0-9]+]] = !{!"llvm.licm.disable"}
// CHECK-DAG: ![[INTERLEAVE_NODE:[0-9]+]] = !{!"llvm.loop.interleave.count", i32 1}
// CHECK-DAG: ![[PIPELINE_DISABLE_NODE:[0-9]+]] = !{!"llvm.loop.pipeline.disable", i1 true}
// CHECK-DAG: ![[II_NODE:[0-9]+]] = !{!"llvm.loop.pipeline.initiationinterval", i32 2}
// CHECK-DAG: ![[ACCESS_GROUPS_NODE:[0-9]+]] = !{![[GROUP_NODE1]], ![[GROUP_NODE2]]}
