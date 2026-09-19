; Regression test: trace-args must NOT emit a non-dominating use. The entry-block #dbg_value
; scan may find an argument whose debug location is a value defined LATER in the function
; (debug records are exempt from SSA dominance). Using such a value as the trace pointer -
; the trace call is inserted at the entry-block terminator - produces IR that fails the
; verifier ("Instruction does not dominate all uses") and, with -disable-llvm-verifier
; (kernel builds), crashes RegisterCoalescer::reMaterializeDef at codegen. Such an argument
; is traced as a null pointer instead. opt runs the verifier, so a successful run of this
; test already proves the emitted IR is well-formed (it would error before this fix).

; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-args -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; %arg1's only debug location is %later, a GEP defined in a later block (does NOT dominate
; the entry terminator). %arg2 is a normal directly-located pointer argument.
define void @arg_loc_not_dominating(ptr %arg1, ptr %arg2) #0 !dbg !8 {
entry:
  #dbg_value(ptr %arg2, !13, !DIExpression(), !15)
  #dbg_value(ptr %later, !12, !DIExpression(), !15)
  br label %bb, !dbg !15
bb:
  %later = getelementptr i8, ptr %arg1, i64 128, !dbg !15
  ret void, !dbg !15
}
; CHECK-LABEL: define void @arg_loc_not_dominating(ptr %arg1, ptr %arg2)
; arg1 (index 0): its debug location did not dominate -> traced as a null pointer, size 0.
; CHECK-DAG: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @arg_loc_not_dominating to i64), i32 0, i32 0, ptr null, ptr {{.*}}, i32 {{.*}})
; arg2 (index 1): a normal pointer arg, traced directly.
; CHECK-DAG: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @arg_loc_not_dominating to i64), i32 1, i32 {{[0-9]+}}, ptr %arg2, ptr {{.*}}, i32 {{.*}})

attributes #0 = { nounwind sanitize_address }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!8 = distinct !DISubprogram(name: "arg_loc_not_dominating", scope: !1, file: !1, line: 1, type: !9, unit: !0, retainedNodes: !11)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !6, !6}
!11 = !{!12, !13}
!12 = !DILocalVariable(name: "arg1", arg: 1, scope: !8, file: !1, line: 1, type: !6)
!13 = !DILocalVariable(name: "arg2", arg: 2, scope: !8, file: !1, line: 1, type: !6)
!15 = !DILocation(line: 1, column: 1, scope: !8)
