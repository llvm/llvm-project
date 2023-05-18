; Test that GCOV instrumentation numbers functions correctly when some
; functions aren't emitted.

; Inject metadata to set the .gcno file location
; RUN: rm -rf %t && mkdir -p %t
; RUN: echo '!14 = !{!"%/t/function-numbering.ll", !0}' > %t/1
; RUN: cat %s %t/1 > %t/2

; RUN: opt -passes=insert-gcov-profiling -S < %t/2 | FileCheck --check-prefix GCDA %s
; RUN: llvm-cov gcov -n -dump %t/function-numbering.gcno 2>&1 | FileCheck --check-prefix GCNO %s
; RUN: opt -passes=insert-gcov-profiling -S < %t/2 -mtriple=s390x-unknown-linux | FileCheck --check-prefix EXT %s
; RUN: opt -passes=insert-gcov-profiling -S < %t/2 -mtriple=mips-linux-gnu | FileCheck --check-prefix MIPS_EXT %s
; REQUIRES: x86-registered-target, systemz-registered-target, mips-registered-target

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; GCDA: @__llvm_internal_gcov_emit_function_args.0 = internal unnamed_addr constant
; GCDA-SAME: { i32 0,
; GCDA-SAME: { i32 1,
;
; GCDA-LABEL: define internal void @__llvm_gcov_writeout() unnamed_addr #[[#ATTR:]] {
; GCDA-NEXT:  entry:
; GCDA-NEXT:    br label %[[FILE_LOOP_HEADER:.*]]
;
; GCDA:       [[FILE_LOOP_HEADER]]:
; GCDA-NEXT:    %[[IV:.*]] = phi i32 [ 0, %entry ], [ %[[NEXT_IV:.*]], %[[FILE_LOOP_LATCH:.*]] ]
; GCDA-NEXT:    %[[FILE_INFO:.*]] = getelementptr inbounds {{.*}}, ptr @__llvm_internal_gcov_emit_file_info, i32 0, i32 %[[IV]]
; GCDA-NEXT:    %[[START_FILE_ARGS:.*]] = getelementptr inbounds {{.*}}, ptr %[[FILE_INFO]], i32 0, i32 0
; GCDA-NEXT:    %[[START_FILE_ARG_0_PTR:.*]] = getelementptr inbounds {{.*}}, ptr %[[START_FILE_ARGS]], i32 0, i32 0
; GCDA-NEXT:    %[[START_FILE_ARG_0:.*]] = load ptr, ptr %[[START_FILE_ARG_0_PTR]]
; GCDA-NEXT:    %[[START_FILE_ARG_1_PTR:.*]] = getelementptr inbounds {{.*}}, ptr %[[START_FILE_ARGS]], i32 0, i32 1
; GCDA-NEXT:    %[[START_FILE_ARG_1:.*]] = load i32, ptr %[[START_FILE_ARG_1_PTR]]
; GCDA-NEXT:    %[[START_FILE_ARG_2_PTR:.*]] = getelementptr inbounds {{.*}}, ptr %[[START_FILE_ARGS]], i32 0, i32 2
; GCDA-NEXT:    %[[START_FILE_ARG_2:.*]] = load i32, ptr %[[START_FILE_ARG_2_PTR]]
; GCDA-NEXT:    call void @llvm_gcda_start_file(ptr %[[START_FILE_ARG_0]], i32 %[[START_FILE_ARG_1]], i32 %[[START_FILE_ARG_2]])
; GCDA-NEXT:    %[[NUM_COUNTERS_PTR:.*]] = getelementptr inbounds {{.*}}, ptr %[[FILE_INFO]], i32 0, i32 1
; GCDA-NEXT:    %[[NUM_COUNTERS:.*]] = load i32, ptr %[[NUM_COUNTERS_PTR]]
; GCDA-NEXT:    %[[EMIT_FUN_ARGS_ARRAY_PTR:.*]] = getelementptr inbounds {{.*}}, ptr %[[FILE_INFO]], i32 0, i32 2
; GCDA-NEXT:    %[[EMIT_FUN_ARGS_ARRAY:.*]] = load ptr, ptr %[[EMIT_FUN_ARGS_ARRAY_PTR]]
; GCDA-NEXT:    %[[EMIT_ARCS_ARGS_ARRAY_PTR:.*]] = getelementptr inbounds {{.*}}, ptr %[[FILE_INFO]], i32 0, i32 3
; GCDA-NEXT:    %[[EMIT_ARCS_ARGS_ARRAY:.*]] = load ptr, ptr %[[EMIT_ARCS_ARGS_ARRAY_PTR]]
; GCDA-NEXT:    %[[ENTER_COUNTER_LOOP_COND:.*]] = icmp slt i32 0, %[[NUM_COUNTERS]]
; GCDA-NEXT:    br i1 %[[ENTER_COUNTER_LOOP_COND]], label %[[COUNTER_LOOP:.*]], label %[[FILE_LOOP_LATCH]]
;
; GCDA:       [[COUNTER_LOOP]]:
; GCDA-NEXT:    %[[JV:.*]] = phi i32 [ 0, %[[FILE_LOOP_HEADER]] ], [ %[[NEXT_JV:.*]], %[[COUNTER_LOOP]] ]
; GCDA-NEXT:    %[[EMIT_FUN_ARGS:.*]] = getelementptr inbounds {{.*}}, ptr %[[EMIT_FUN_ARGS_ARRAY]], i32 %[[JV]]
; GCDA-NEXT:    %[[EMIT_FUN_ARG_0_PTR:.*]] = getelementptr inbounds {{.*}}, ptr %[[EMIT_FUN_ARGS]], i32 0, i32 0
; GCDA-NEXT:    %[[EMIT_FUN_ARG_0:.*]] = load i32, ptr %[[EMIT_FUN_ARG_0_PTR]]
; GCDA-NEXT:    %[[EMIT_FUN_ARG_1_PTR:.*]] = getelementptr inbounds {{.*}}, ptr %[[EMIT_FUN_ARGS]], i32 0, i32 1
; GCDA-NEXT:    %[[EMIT_FUN_ARG_1:.*]] = load i32, ptr %[[EMIT_FUN_ARG_1_PTR]]
; GCDA-NEXT:    %[[EMIT_FUN_ARG_2_PTR:.*]] = getelementptr inbounds {{.*}}, ptr %[[EMIT_FUN_ARGS]], i32 0, i32 2
; GCDA-NEXT:    %[[EMIT_FUN_ARG_2:.*]] = load i32, ptr %[[EMIT_FUN_ARG_2_PTR]]
; GCDA-NEXT:    call void @llvm_gcda_emit_function(i32 %[[EMIT_FUN_ARG_0]],
; GCDA-SAME:                                       i32 %[[EMIT_FUN_ARG_1]],
; GCDA-SAME:                                       i32 %[[EMIT_FUN_ARG_2]])
; GCDA-NEXT:    %[[EMIT_ARCS_ARGS:.*]] = getelementptr inbounds {{.*}}, ptr %[[EMIT_ARCS_ARGS_ARRAY]], i32 %[[JV]]
; GCDA-NEXT:    %[[EMIT_ARCS_ARG_0_PTR:.*]] = getelementptr inbounds {{.*}}, ptr %[[EMIT_ARCS_ARGS]], i32 0, i32 0
; GCDA-NEXT:    %[[EMIT_ARCS_ARG_0:.*]] = load i32, ptr %[[EMIT_ARCS_ARG_0_PTR]]
; GCDA-NEXT:    %[[EMIT_ARCS_ARG_1_PTR:.*]] = getelementptr inbounds {{.*}}, ptr %[[EMIT_ARCS_ARGS]], i32 0, i32 1
; GCDA-NEXT:    %[[EMIT_ARCS_ARG_1:.*]] = load ptr, ptr %[[EMIT_ARCS_ARG_1_PTR]]
; GCDA-NEXT:    call void @llvm_gcda_emit_arcs(i32 %[[EMIT_ARCS_ARG_0]],
; GCDA-SAME:                                   ptr %[[EMIT_ARCS_ARG_1]])
; GCDA-NEXT:    %[[NEXT_JV]] = add i32 %[[JV]], 1
; GCDA-NEXT:    %[[COUNTER_LOOP_COND:.*]] = icmp slt i32 %[[NEXT_JV]], %[[NUM_COUNTERS]]
; GCDA-NEXT:    br i1 %[[COUNTER_LOOP_COND]], label %[[COUNTER_LOOP]], label %[[FILE_LOOP_LATCH]]
;
; GCDA:       [[FILE_LOOP_LATCH]]:
; GCDA-NEXT:    call void @llvm_gcda_summary_info()
; GCDA-NEXT:    call void @llvm_gcda_end_file()
; GCDA-NEXT:    %[[NEXT_IV]] = add i32 %[[IV]], 1
; GCDA-NEXT:    %[[FILE_LOOP_COND:.*]] = icmp slt i32 %[[NEXT_IV]], 1
; GCDA-NEXT:    br i1 %[[FILE_LOOP_COND]], label %[[FILE_LOOP_HEADER]], label %[[EXIT:.*]]
;
; GCDA:       [[EXIT]]:
; GCDA-NEXT:    ret void

; GCNO: == foo (0) @
; GCNO-NOT: == bar ({{[0-9]+}}) @
; GCNO: == baz (1) @

; GCDA:     declare void @llvm_gcda_start_file(ptr, i32, i32)
; EXT:      declare void @llvm_gcda_start_file(ptr, i32 zeroext, i32 zeroext)
; MIPS_EXT: declare void @llvm_gcda_start_file(ptr, i32 signext, i32 signext)

; GCDA:     declare void @llvm_gcda_emit_function(i32, i32, i32)
; EXT:      declare void @llvm_gcda_emit_function(i32 zeroext, i32 zeroext, i32 zeroext)
; MIPS_EXT: declare void @llvm_gcda_emit_function(i32 signext, i32 signext, i32 signext)

; GCDA:     declare void @llvm_gcda_emit_arcs(i32, ptr)
; EXT:      declare void @llvm_gcda_emit_arcs(i32 zeroext, ptr)
; MIPS_EXT: declare void @llvm_gcda_emit_arcs(i32 signext, ptr)

define void @foo() !dbg !4 {
  ret void, !dbg !12
}

define void @bar() !dbg !7 {
  ; This function is referenced by the debug info, but no lines have locations.
  ret void
}

define void @baz() !dbg !8 {
  ret void, !dbg !13
}

; GCDA: attributes #[[#ATTR]] = { noinline nounwind }

!llvm.gcov = !{!14}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0 ", isOptimized: false, emissionKind: LineTablesOnly, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: ".../llvm/test/Transforms/GCOVProfiling/function-numbering.ll", directory: "")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: ".../llvm/test/Transforms/GCOVProfiling/function-numbering.ll", directory: "")
!6 = !DISubroutineType(types: !2)
!7 = distinct !DISubprogram(name: "bar", line: 2, isLocal: false, isDefinition: true, isOptimized: false, unit: !0, scopeLine: 2, file: !1, scope: !5, type: !6, retainedNodes: !2)
!8 = distinct !DISubprogram(name: "baz", line: 3, isLocal: false, isDefinition: true, isOptimized: false, unit: !0, scopeLine: 3, file: !1, scope: !5, type: !6, retainedNodes: !2)
!9 = !{i32 2, !"Dwarf Version", i32 2}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.6.0 "}
!12 = !DILocation(line: 1, column: 13, scope: !4)
!13 = !DILocation(line: 3, column: 13, scope: !8)
