; Test that we annotate entire program's summary to IR.
; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/summary.prof -S | FileCheck %s

define i32 @bar() #0 !dbg !1 {
entry:
  ret i32 1, !dbg !2
}

; CHECK-DAG: {{![0-9]+}} = !{i32 1, !"ProfileSummary", {{![0-9]+}}}
; CHECK-DAG: {{![0-9]+}} = !{!"NumFunctions", i64 2}
; CHECK-DAG: {{![0-9]+}} = !{!"MaxFunctionCount", i64 3}

!1 = distinct !DISubprogram(name: "bar")
!2 = !DILocation(line: 2, scope: !2)
