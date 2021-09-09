; The purpose of this test is to check that the remark emission analysis result
; stays valid when it can and is invalidated otherwise. The code is just
; a minimal amount of code to exercise the pass.
;
; First make sure we emit remarks on this test case.
; RUN: opt %s -disable-output -aa-pipeline=basic-aa 2>&1 \
; RUN:     -passes='require<opt-remark-emit>,loop-mssa(licm)' \
; RUN:     -pass-remarks=licm -pass-remarks-with-hotness \
; RUN:     | FileCheck %s
;
; Check that passes which preserve BFI don't invalidate the emitter.
; RUN: opt %s -disable-output -aa-pipeline=basic-aa 2>&1 \
; RUN:     -passes='require<opt-remark-emit>,instcombine,require<opt-remark-emit>,loop-mssa(licm)' -debug-pass-manager \
; RUN:     -pass-remarks=licm -pass-remarks-with-hotness \
; RUN:     | FileCheck %s --check-prefixes=CHECK,CHECK-PM-PRESERVE
;
; Check that invalidating BFI computes a fresh emitter.
; RUN: opt %s -disable-output -aa-pipeline=basic-aa 2>&1 \
; RUN:     -passes='require<opt-remark-emit>,invalidate<block-freq>,require<opt-remark-emit>,loop-mssa(licm)' -debug-pass-manager \
; RUN:     -pass-remarks=licm -pass-remarks-with-hotness \
; RUN:     | FileCheck %s --check-prefixes=CHECK,CHECK-PM-INVALIDATE
;
; Check that invalidating BFI desn't compute a fresh emitter when we don't
; request hotness remarks.
; RUN: opt %s -disable-output -aa-pipeline=basic-aa 2>&1 \
; RUN:     -passes='require<opt-remark-emit>,invalidate<block-freq>,require<opt-remark-emit>,loop-mssa(licm)' -debug-pass-manager \
; RUN:     -pass-remarks=licm \
; RUN:     | FileCheck %s --check-prefixes=CHECK,CHECK-PM-NO-INVALIDATE

target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

define void @hoist(i32* %array, i32* noalias %p) {
; CHECK-PM-PRESERVE: Running analysis: OptimizationRemarkEmitterAnalysis
; CHECK-PM-PRESERVE: Running pass: InstCombinePass
; CHECK-PM-PRESERVE-NOT: Invalidating analysis: OptimizationRemarkEmitterAnalysis
; CHECK-PM-PRESERVE-NOT: Running analysis: OptimizationRemarkEmitterAnalysis
; CHECK-PM-PRESERVE: Running pass: LICMPass
; CHECK-PM-INVALIDATE: Running analysis: OptimizationRemarkEmitterAnalysis
; CHECK-PM-INVALIDATE: Running pass: InvalidateAnalysisPass
; CHECK-PM-INVALIDATE: Invalidating analysis: OptimizationRemarkEmitterAnalysis
; CHECK-PM-INVALIDATE: Running analysis: OptimizationRemarkEmitterAnalysis
; CHECK-PM-INVALIDATE: Running pass: LICMPass
; CHECK-PM-NO-INVALIDATE: Running analysis: OptimizationRemarkEmitterAnalysis
; CHECK-PM-NO-INVALIDATE: Running pass: InvalidateAnalysisPass
; CHECK-PM-NO-INVALIDATE-NOT: Invalidating analysis: OptimizationRemarkEmitterAnalysis
; CHECK-PM-NO-INVALIDATE-NOT: Running analysis: OptimizationRemarkEmitterAnalysis
; CHECK-PM-NO-INVALIDATE: Running pass: LICMPass

Entry:
  br label %Loop

Loop:
  %j = phi i32 [ 0, %Entry ], [ %Next, %Loop ]
  %addr = getelementptr i32, i32* %array, i32 %j
  %a = load i32, i32* %addr
; CHECK: remark: /tmp/kk.c:2:20: hoisting load
  %b = load i32, i32* %p, !dbg !8
  %a2 = add i32 %a, %b
  store i32 %a2, i32* %addr
  %Next = add i32 %j, 1
  %cond = icmp eq i32 %Next, 0
  br i1 %cond, label %Out, label %Loop

Out:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/kk.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"PIC Level", i32 2}
!5 = !{!"clang version 3.9.0 "}
!6 = distinct !DISubprogram(name: "success", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 2, column: 20, scope: !6)
