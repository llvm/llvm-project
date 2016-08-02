; Test that GCOV instrumentation numbers functions correctly when some
; functions aren't emitted.

; Inject metadata to set the .gcno file location
; RUN: echo '!14 = !{!"%/T/function-numbering.ll", !0}' > %t1
; RUN: cat %s %t1 > %t2

; RUN: opt -insert-gcov-profiling -S < %t2 | FileCheck --check-prefix GCDA %s
; RUN: llvm-cov gcov -n -dump %T/function-numbering.gcno 2>&1 | FileCheck --check-prefix GCNO %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; GCDA: @[[FOO:[0-9]+]] = private unnamed_addr constant [4 x i8] c"foo\00"
; GCDA-NOT: @{{[0-9]+}} = private unnamed_addr constant .* c"bar\00"
; GCDA: @[[BAZ:[0-9]+]] = private unnamed_addr constant [4 x i8] c"baz\00"
; GCDA: define internal void @__llvm_gcov_writeout()
; GCDA: call void @llvm_gcda_emit_function(i32 0, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @[[FOO]]
; GCDA: call void @llvm_gcda_emit_function(i32 1, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @[[BAZ]]

; GCNO: == foo (0) @
; GCNO-NOT: == bar ({{[0-9]+}}) @
; GCNO: == baz (1) @

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

!llvm.gcov = !{!14}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0 ", isOptimized: false, emissionKind: 2, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: ".../llvm/test/Transforms/GCOVProfiling/function-numbering.ll", directory: "")
!2 = !{}
!3 = !{!4, !7, !8}
!4 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, isOptimized: false, scopeLine: 1, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: ".../llvm/test/Transforms/GCOVProfiling/function-numbering.ll", directory: "")
!6 = !DISubroutineType(types: !2)
!7 = distinct !DISubprogram(name: "bar", line: 2, isLocal: false, isDefinition: true, isOptimized: false, scopeLine: 2, file: !1, scope: !5, type: !6, variables: !2)
!8 = distinct !DISubprogram(name: "baz", line: 3, isLocal: false, isDefinition: true, isOptimized: false, scopeLine: 3, file: !1, scope: !5, type: !6, variables: !2)
!9 = !{i32 2, !"Dwarf Version", i32 2}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{!"clang version 3.6.0 "}
!12 = !DILocation(line: 1, column: 13, scope: !4)
!13 = !DILocation(line: 3, column: 13, scope: !8)
