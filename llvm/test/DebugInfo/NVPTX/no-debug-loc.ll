; RUN: split-file %s %t
; RUN: llc < %t/debug-then-nodebug.ll -mtriple=nvptx64-nvidia-cuda | FileCheck %t/debug-then-nodebug.ll
; RUN: llc < %t/nodebug-then-debug.ll -mtriple=nvptx64-nvidia-cuda | FileCheck %t/nodebug-then-debug.ll
; RUN: %if ptxas %{ llc < %t/debug-then-nodebug.ll -mtriple=nvptx64-nvidia-cuda | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %t/nodebug-then-debug.ll -mtriple=nvptx64-nvidia-cuda | %ptxas-verify %}

;; When a module contains multiple CUs where one is DebugDirectivesOnly and
;; another is NoDebug, we would attempt to emit dwarf directives for the
;; NoDebug CU leading to an assertion. This test verifies that we only emit
;; dwarf directives for the DebugDirectivesOnly CU regardless of function order.

;--- debug-then-nodebug.ll
define i32 @foo(i32 %a, i32 %b) !dbg !5 {
; CHECK-LABEL: foo
; CHECK:     .loc    [[FILE:[0-9]+]] 26 0          // debug_directives_only.cu:26:0
; CHECK-NOT: .loc    [[FILE]]        26 0          // debug_directives_only.cu:26:0
; CHECK:     .loc    [[FILE]]        40 22         // debug_directives_only.cu:40:22

  %add = add i32 %b, %a, !dbg !8
  ret i32 %add, !dbg !8
}

define i32 @bar(i32 %a, i32 %b) !dbg !40 {
; CHECK-LABEL: bar
; CHECK-NOT: .loc

  %add = add i32 %b, %a, !dbg !41
  ret i32 %add, !dbg !41
}

; CHECK:     .file   [[FILE]] "/test/directory/debug_directives_only.cu"
; CHECK-NOT: .file   {{[0-9]*}} "/test/directory/no_debug.cu"
; CHECK-NOT: .section .debug{{.*}}

!llvm.dbg.cu = !{!0, !2}
!nvvm.annotations = !{}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: DebugDirectivesOnly)
!1 = !DIFile(filename: "debug_directives_only.cu", directory: "/test/directory/")
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug)
!3 = !DIFile(filename: "no_debug.cu", directory: "/test/directory/")
!4 = !{i32 1, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "kernel", linkageName: "foo", scope: !1, file: !1, line: 123, type: !6, scopeLine: 26, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 40, column: 22, scope: !31)
!31 = distinct !DILexicalBlock(scope: !5, file: !1, line: 3, column: 17)
!40 = distinct !DISubprogram(name: "kernel", linkageName: "bar", scope: !3, file: !3, line: 234, type: !6, scopeLine: 26, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!41 = !DILocation(line: 40, column: 22, scope: !42)
!42 = distinct !DILexicalBlock(scope: !40, file: !3, line: 6, column: 8)

;--- nodebug-then-debug.ll
define i32 @baz(i32 %a, i32 %b) !dbg !5 {
; CHECK-LABEL: baz
; CHECK-NOT: .loc

  %add = add i32 %b, %a, !dbg !8
  ret i32 %add, !dbg !8
}

define i32 @qux(i32 %a, i32 %b) !dbg !40 {
; CHECK-LABEL: qux
; CHECK:     .loc    [[FILE:[0-9]+]] 26 0          // debug_directives_only.cu:26:0
; CHECK-NOT: .loc    [[FILE]]        26 0          // debug_directives_only.cu:26:0
; CHECK:     .loc    [[FILE]]        40 22         // debug_directives_only.cu:40:22

  %add = add i32 %b, %a, !dbg !41
  ret i32 %add, !dbg !41
}

; CHECK:     .file   [[FILE]] "/test/directory/debug_directives_only.cu"
; CHECK-NOT: .file   {{[0-9]*}} "/test/directory/no_debug.cu"
; CHECK-NOT: .section .debug{{.*}}

!llvm.dbg.cu = !{!0, !2}
!nvvm.annotations = !{}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "no_debug.cu", directory: "/test/directory/")
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: DebugDirectivesOnly)
!3 = !DIFile(filename: "debug_directives_only.cu", directory: "/test/directory/")
!4 = !{i32 1, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "kernel", linkageName: "baz", scope: !1, file: !1, line: 345, type: !6, scopeLine: 26, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 40, column: 22, scope: !31)
!31 = distinct !DILexicalBlock(scope: !5, file: !1, line: 6, column: 8)
!40 = distinct !DISubprogram(name: "kernel", linkageName: "qux", scope: !3, file: !3, line: 456, type: !6, scopeLine: 26, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!41 = !DILocation(line: 40, column: 22, scope: !42)
!42 = distinct !DILexicalBlock(scope: !40, file: !3, line: 3, column: 17)
