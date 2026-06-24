; -combiner-disabled suppresses DAGCombine's generic visit() pass.  We need it
; here because DAGCombine's reassociateOpsCommutative fires unconditionally at
; -O0 and would fold (a+42) + (a+42) into (a+a) + 84, hiding the per-line .loc
; directives this test asserts on.  This test exercises NVPTXSDNodeCSEMap in
; isolation; DAGCombine's behavior at -O0 is a separate, pre-existing concern.
;
; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -O0 -combiner-disabled | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64-nvidia-cuda -O0 -combiner-disabled | %ptxas-verify %}

; At -O0, NVPTXSDNodeCSEMap excludes ISD::Constant from DebugLoc keying, so
; identical constant SDNodes are always shared regardless of source location.
; Operations with constant operands therefore inherit the normal DebugLoc-keyed
; behavior: same DebugLoc folds, different DebugLoc does not.

; -- Same DebugLoc: identical operations sharing a constant operand must be
;    folded into one instruction; the consumer must see the same register for
;    both operands.
define i32 @const_same_loc(i32 %a) !dbg !3 {
; CHECK-LABEL: const_same_loc(
; CHECK:     .loc {{[0-9]+}} 1
; CHECK:     add.s32 [[REG:%r[0-9]+]],
; CHECK-NOT: .loc {{[0-9]+}} 1
; CHECK:     .loc {{[0-9]+}} 3
; CHECK:     add.s32 {{%r[0-9]+}}, [[REG]], [[REG]]
  %x = add i32 %a, 42, !dbg !6
  %y = add i32 %a, 42, !dbg !6
  %z = add i32 %x, %y, !dbg !8
  ret i32 %z, !dbg !9
}

; -- Different DebugLoc: identical operations sharing a constant operand must
;    each produce their own .loc and instruction, even though the constant
;    operand 42 is one shared SDNode.
define i32 @const_diff_loc(i32 %a) !dbg !10 {
; CHECK-LABEL: const_diff_loc(
; CHECK: .loc {{[0-9]+}} 1
; CHECK: add.s32
; CHECK: .loc {{[0-9]+}} 2
; CHECK: add.s32
  %x = add i32 %a, 42, !dbg !11
  %y = add i32 %a, 42, !dbg !12
  %z = add i32 %x, %y, !dbg !13
  ret i32 %z, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cu", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "const_same_loc", scope: !1, file: !1, line: 1, type: !4, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !5)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !DILocation(line: 1, column: 1, scope: !3)
!8 = !DILocation(line: 3, column: 1, scope: !3)
!9 = !DILocation(line: 4, column: 1, scope: !3)
!10 = distinct !DISubprogram(name: "const_diff_loc", scope: !1, file: !1, line: 1, type: !4, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !5)
!11 = !DILocation(line: 1, column: 1, scope: !10)
!12 = !DILocation(line: 2, column: 1, scope: !10)
!13 = !DILocation(line: 3, column: 1, scope: !10)
!14 = !DILocation(line: 4, column: 1, scope: !10)
