; RUN: llc -mtriple=x86_64-apple-darwin -filetype=obj < %s -o %t.o
; RUN: llvm-dwarfdump %t.o | FileCheck %s -implicit-check-not=DW_TAG_subprogram
; RUN: llvm-dwarfdump --verify %t.o

; This test checks that cross-CU references within call site tags to subprogram
; definitions are well-formed. There are 5 cases checked in this test. Each set
; of checks is numbered and has a brief summary.

; Instructions to regenerate the IR:
; clang -O1 -g -emit-llvm -o a.bc -c a.c
; clang -O1 -g -emit-llvm -o b.bc -c b.c
; llvm-link -o linked.bc a.bc b.bc
; opt -O1 linked.bc -o merged.bc

; Source:
; // a.c
; __attribute__((optnone)) void noinline_func_in_a() {}
;
; __attribute__((optnone)) static void foo() {}
; __attribute__((always_inline)) void always_inline_helper_in_a_that_calls_foo() {
;   foo();
; }
;
; extern void func_from_b();
; void call_func_in_b_from_a() {
;   func_from_b();
; }
;
; // b.c
; extern void noinline_func_in_a();
; void call_noinline_func_in_a_from_b() {
;   noinline_func_in_a();
; }
;
; __attribute__((optnone)) void foo() {}
; extern void always_inline_helper_in_a_that_calls_foo();
; void call_both_foos_from_b() {
;   foo();
;   always_inline_helper_in_a_that_calls_foo();
; }
;
; __attribute__((optnone)) void func_from_b() {}
; void call_func_in_b_from_b() {
;   func_from_b();
; }

; === CU for a.c ===

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_AT_name ("a.c")

; CHECK: 0x{{0+}}[[NOINLINE_FUNC_IN_A:.*]]: DW_TAG_subprogram
; CHECK:   DW_AT_name ("noinline_func_in_a")

; 1) Check that "always_inline_helper_in_a_that_calls_foo" calls the "foo" in
; a.c, and *not* the "foo" in b.c.
; CHECK: 0x{{0+}}[[ALWAYS_INLINE_HELPER_IN_A:.*]]: DW_TAG_subprogram
; CHECK:   DW_AT_abstract_origin ({{.*}} "always_inline_helper_in_a_that_calls_foo")
; CHECK:   DW_TAG_call_site
; CHECK-NEXT: DW_AT_call_origin (0x{{0+}}[[FOO_IN_A:.*]])

; CHECK: 0x{{0+}}[[FOO_IN_A]]: DW_TAG_subprogram
; CHECK:   DW_AT_name ("foo")

; 2) Check that "call_func_in_b_from_a" has a cross-CU ref into b.c.
; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_name ("call_func_in_b_from_a")
; CHECK:   DW_TAG_call_site
; CHECK-NEXT: DW_AT_call_origin (0x{{0+}}[[FUNC_FROM_B:.*]])

; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_name ("always_inline_helper_in_a_that_calls_foo")
; CHECK:   DW_AT_inline (DW_INL_inlined)

; === CU for b.c ===

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_AT_name ("b.c")

; 3) Validate the cross-CU ref from "call_func_in_b_from_a" in a.c.
; CHECK: 0x{{0+}}[[FUNC_FROM_B]]: DW_TAG_subprogram
; CHECK:   DW_AT_name ("func_from_b")

; 4) Validate the cross-CU ref from "call_noinline_func_in_a_from_b" in b.c.
; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_name ("call_noinline_func_in_a_from_b")
; CHECK:   DW_TAG_call_site
; CHECK-NEXT: DW_AT_call_origin (0x{{0+}}[[NOINLINE_FUNC_IN_A]])

; CHECK: 0x{{0+}}[[FOO_IN_B:.*]]: DW_TAG_subprogram
; CHECK:   DW_AT_name ("foo")

; 5) Validate that we correctly emit a cross-CU ref when the call is inlined
; from another CU.
; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_name ("call_both_foos_from_b")
; CHECK:   DW_TAG_call_site
; CHECK-NEXT: DW_AT_call_origin (0x{{0+}}[[FOO_IN_B]])
; CHECK:   DW_TAG_call_site
; CHECK-NEXT: DW_AT_call_origin (0x{{0+}}[[FOO_IN_A]])

; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_name ("call_func_in_b_from_b")
; CHECK:   DW_TAG_call_site
; CHECK-NEXT: DW_AT_call_origin (0x{{0+}}[[FUNC_FROM_B]])

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

define void @noinline_func_in_a() local_unnamed_addr #0 !dbg !17 {
entry:
  ret void, !dbg !20
}

define void @always_inline_helper_in_a_that_calls_foo() local_unnamed_addr #1 !dbg !21 {
entry:
  tail call fastcc void @foo.2(), !dbg !22
  ret void, !dbg !23
}

define internal fastcc void @foo.2() unnamed_addr #0 !dbg !24 {
entry:
  ret void, !dbg !25
}

define void @call_func_in_b_from_a() local_unnamed_addr !dbg !26 {
entry:
  tail call void @func_from_b() #3, !dbg !27
  ret void, !dbg !28
}

define void @call_noinline_func_in_a_from_b() local_unnamed_addr !dbg !29 {
entry:
  tail call void @noinline_func_in_a() #3, !dbg !30
  ret void, !dbg !31
}

define void @foo() local_unnamed_addr #0 !dbg !32 {
entry:
  ret void, !dbg !33
}

define void @call_both_foos_from_b() local_unnamed_addr !dbg !34 {
entry:
  tail call void @foo(), !dbg !35
  tail call fastcc void @foo.2() #3, !dbg !36
  ret void, !dbg !38
}

define void @func_from_b() local_unnamed_addr #0 !dbg !39 {
entry:
  ret void, !dbg !40
}

define void @call_func_in_b_from_b() local_unnamed_addr !dbg !41 {
entry:
  tail call void @func_from_b(), !dbg !42
  ret void, !dbg !43
}

attributes #0 = { noinline }
attributes #1 = { alwaysinline }

!llvm.dbg.cu = !{!0, !7}
!llvm.ident = !{!12, !12}
!llvm.module.flags = !{!13, !14, !15, !16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (git@github.com:llvm/llvm-project.git 310e85309f870ee7347ef979d7d8da9bf28e92ea)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "a.c", directory: "/Users/vsk/tmp/lto-entry-vals")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "func_from_b", scope: !1, file: !1, line: 8, type: !5, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null, null}
!7 = distinct !DICompileUnit(language: DW_LANG_C99, file: !8, producer: "clang version 10.0.0 (git@github.com:llvm/llvm-project.git 310e85309f870ee7347ef979d7d8da9bf28e92ea)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !9, nameTableKind: None)
!8 = !DIFile(filename: "b.c", directory: "/Users/vsk/tmp/lto-entry-vals")
!9 = !{!10, !11}
!10 = !DISubprogram(name: "noinline_func_in_a", scope: !8, file: !8, line: 1, type: !5, spFlags: DISPFlagOptimized, retainedNodes: !2)
!11 = !DISubprogram(name: "always_inline_helper_in_a_that_calls_foo", scope: !8, file: !8, line: 7, type: !5, spFlags: DISPFlagOptimized, retainedNodes: !2)
!12 = !{!"clang version 10.0.0 (git@github.com:llvm/llvm-project.git 310e85309f870ee7347ef979d7d8da9bf28e92ea)"}
!13 = !{i32 7, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{i32 7, !"PIC Level", i32 2}
!17 = distinct !DISubprogram(name: "noinline_func_in_a", scope: !1, file: !1, line: 1, type: !18, scopeLine: 1, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!18 = !DISubroutineType(types: !19)
!19 = !{null}
!20 = !DILocation(line: 1, column: 53, scope: !17)
!21 = distinct !DISubprogram(name: "always_inline_helper_in_a_that_calls_foo", scope: !1, file: !1, line: 4, type: !18, scopeLine: 4, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!22 = !DILocation(line: 5, column: 3, scope: !21)
!23 = !DILocation(line: 6, column: 1, scope: !21)
!24 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !18, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!25 = !DILocation(line: 3, column: 45, scope: !24)
!26 = distinct !DISubprogram(name: "call_func_in_b_from_a", scope: !1, file: !1, line: 9, type: !18, scopeLine: 9, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!27 = !DILocation(line: 10, column: 3, scope: !26)
!28 = !DILocation(line: 11, column: 1, scope: !26)
!29 = distinct !DISubprogram(name: "call_noinline_func_in_a_from_b", scope: !8, file: !8, line: 2, type: !18, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !7, retainedNodes: !2)
!30 = !DILocation(line: 3, column: 3, scope: !29)
!31 = !DILocation(line: 4, column: 1, scope: !29)
!32 = distinct !DISubprogram(name: "foo", scope: !8, file: !8, line: 6, type: !18, scopeLine: 6, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !7, retainedNodes: !2)
!33 = !DILocation(line: 6, column: 38, scope: !32)
!34 = distinct !DISubprogram(name: "call_both_foos_from_b", scope: !8, file: !8, line: 8, type: !18, scopeLine: 8, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !7, retainedNodes: !2)
!35 = !DILocation(line: 9, column: 3, scope: !34)
!36 = !DILocation(line: 5, column: 3, scope: !21, inlinedAt: !37)
!37 = distinct !DILocation(line: 10, column: 3, scope: !34)
!38 = !DILocation(line: 11, column: 1, scope: !34)
!39 = distinct !DISubprogram(name: "func_from_b", scope: !8, file: !8, line: 13, type: !18, scopeLine: 13, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !7, retainedNodes: !2)
!40 = !DILocation(line: 13, column: 46, scope: !39)
!41 = distinct !DISubprogram(name: "call_func_in_b_from_b", scope: !8, file: !8, line: 14, type: !18, scopeLine: 14, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !7, retainedNodes: !2)
!42 = !DILocation(line: 15, column: 3, scope: !41)
!43 = !DILocation(line: 16, column: 1, scope: !41)
