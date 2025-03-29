// RUN: mlir-opt %s --split-input-file | mlir-opt --split-input-file | FileCheck %s

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

// CHECK-DAG: #[[PEELED:.*]] = #llvm.loop_peeled<count = 8 : i32>
#peeled = #llvm.loop_peeled<count = 8 : i32>

// CHECK-DAG: #[[UNSWITCH:.*]] = #llvm.loop_unswitch<partialDisable = true>
#unswitch = #llvm.loop_unswitch<partialDisable = true>

// CHECK-DAG: #[[GROUP1:.*]] = #llvm.access_group<id = {{.*}}>
// CHECK-DAG: #[[GROUP2:.*]] = #llvm.access_group<id = {{.*}}>
#group1 = #llvm.access_group<id = distinct[0]<>>
#group2 = #llvm.access_group<id = distinct[1]<>>

// CHECK: #[[LOOP_ANNOT:.*]] = #llvm.loop_annotation<
// CHECK-DAG: disableNonforced = false
// CHECK-DAG: mustProgress = true
// CHECK-DAG: unroll = #[[UNROLL]]
// CHECK-DAG: unrollAndJam = #[[UNROLL_AND_JAM]]
// CHECK-DAG: licm = #[[LICM]]
// CHECK-DAG: distribute = #[[DISTRIBUTE]]
// CHECK-DAG: pipeline = #[[PIPELINE]]
// CHECK-DAG: peeled = #[[PEELED]]
// CHECK-DAG: unswitch = #[[UNSWITCH]]
// CHECK-DAG: isVectorized = false
// CHECK-DAG: parallelAccesses = #[[GROUP1]], #[[GROUP2]]>
#loopMD = #llvm.loop_annotation<disableNonforced = false,
        mustProgress = true,
        vectorize = #vectorize,
        interleave = #interleave,
        unroll = #unroll,
        unrollAndJam = #unrollAndJam,
        licm = #licm,
        distribute = #distribute,
        pipeline = #pipeline,
        peeled = #peeled,
        unswitch = #unswitch,
        isVectorized = false,
        parallelAccesses = #group1, #group2>

// CHECK: llvm.func @loop_annotation
llvm.func @loop_annotation() {
  // CHECK: llvm.br ^bb1 {llvm.loop = #[[LOOP_ANNOT]]
  llvm.br ^bb1 {llvm.loop = #loopMD}
^bb1:
  llvm.return
}

// -----

#di_file = #llvm.di_file<"metadata-loop.ll" in "/">

// CHECK-DAG: #[[START_LOC:.*]] = loc("loop-metadata.mlir":42:4)
#loc1 = loc("loop-metadata.mlir":42:4)
// CHECK-DAG: #[[END_LOC:.*]] = loc("loop-metadata.mlir":52:4)
#loc2 = loc("loop-metadata.mlir":52:4)

#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file, isOptimized = false, emissionKind = None>
// CHECK-DAG: #[[SUBPROGRAM:.*]] = #llvm.di_subprogram<
#di_subprogram = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "loop_locs", file = #di_file, subprogramFlags = Definition>

// CHECK-DAG: #[[START_LOC_FUSED:.*]] = loc(fused<#[[SUBPROGRAM]]>[#[[START_LOC]]]
#start_loc_fused = loc(fused<#di_subprogram>[#loc1])
// CHECK-DAG: #[[END_LOC_FUSED:.*]] = loc(fused<#[[SUBPROGRAM]]>[#[[END_LOC]]]
#end_loc_fused= loc(fused<#di_subprogram>[#loc2])

// CHECK-DAG: #[[GROUP1:.*]] = #llvm.access_group<id = {{.*}}>
// CHECK-DAG: #[[GROUP2:.*]] = #llvm.access_group<id = {{.*}}>
#group1 = #llvm.access_group<id = distinct[0]<>>
#group2 = #llvm.access_group<id = distinct[1]<>>

// CHECK: #[[LOOP_ANNOT:.*]] = #llvm.loop_annotation<
// CHECK-DAG: disableNonforced = false
// CHECK-DAG: startLoc = #[[START_LOC_FUSED]]
// CHECK-DAG: endLoc = #[[END_LOC_FUSED]]
// CHECK-DAG: parallelAccesses = #[[GROUP1]], #[[GROUP2]]>
#loopMD = #llvm.loop_annotation<disableNonforced = false,
        mustProgress = true,
        startLoc = #start_loc_fused,
        endLoc = #end_loc_fused,
        parallelAccesses = #group1, #group2>

// CHECK: llvm.func @loop_annotation_with_locs
llvm.func @loop_annotation_with_locs() {
  // CHECK: llvm.br ^bb1 {loop_annotation = #[[LOOP_ANNOT]]
  llvm.br ^bb1 {loop_annotation = #loopMD}
^bb1:
  llvm.return
}
