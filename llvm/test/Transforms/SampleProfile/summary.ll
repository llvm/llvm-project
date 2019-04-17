; Test that we annotate entire program's summary to IR.
; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/summary.prof -S | FileCheck %s
; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/summary.prof -S | opt -sample-profile -sample-profile-file=%S/Inputs/summary.prof -S | FileCheck %s

define i32 @bar() #0 !dbg !1 {
entry:
  ret i32 1, !dbg !2
}

define i32 @baz() !dbg !3 {
entry:
    %call = call i32 @bar(), !dbg !4
    ret i32 %call, !dbg !5
}

; CHECK-DAG: {{![0-9]+}} = !{i32 1, !"ProfileSummary", {{![0-9]+}}}
; CHECK-DAG: {{![0-9]+}} = !{!"TotalCount", i64 900}
; CHECK-DAG: {{![0-9]+}} = !{!"NumCounts", i64 5}
; CHECK-DAG: {{![0-9]+}} = !{!"NumFunctions", i64 3}
; CHECK-DAG: {{![0-9]+}} = !{!"MaxFunctionCount", i64 3}

!1 = distinct !DISubprogram(name: "bar")
!2 = !DILocation(line: 2, scope: !2)
!3 = distinct !DISubprogram(name: "baz")
!4 = !DILocation(line: 1, scope: !4)
!5 = !DILocation(line: 2, scope: !5)
