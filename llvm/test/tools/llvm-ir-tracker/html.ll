; RUN: rm -f %t.tsv %t.db
; RUN: printf 'P\t0\tinitial\t<initial>\tf\nT\t1\t/tmp/show.c\t8\t3\nI\tf\tentry\t0\tadd\t1\tadd i32 1, 2\nT\t2\t/tmp/show.c\t9\t3\nI\tf\tentry\t1\tret\t2\tret i32 3\nP\t1\tafter\tinstcombine\tf\nI\tf\tentry\t0\tadd\t1\tadd i32 1, 3\n' > %t.tsv
; RUN: %ir-tracker build --input %t.tsv --db %t.db
; RUN: rm -rf %t.html
; RUN: %ir-tracker html --db %t.db -o %t.html --no-highlight | FileCheck %s --check-prefix=LOG
; RUN: ls %t.html | FileCheck %s --check-prefix=FILES
; RUN: FileCheck %s --check-prefix=INDEX --input-file=%t.html/index.html
; RUN: FileCheck %s --check-prefix=PAGE --input-file=%t.html/fn-f.html

define i32 @f(i32 %x) !dbg !6 {
entry:
  %a = add i32 %x, 1, !dbg !8
  ret i32 %a, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "ir-tracker-test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "show.c", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!4 = !DISubroutineType(types: !5)
!5 = !{!3, !3}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 7, type: !4, scopeLine: 7, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 8, column: 3, scope: !6)
!9 = !DILocation(line: 9, column: 3, scope: !6)

; LOG: ir-tracker: wrote 1 function page(s) + index

; FILES-DAG: index.html
; FILES-DAG: fn-f.html
; FILES-DAG: style.css

; INDEX: ir-tracker report
; INDEX: show.c
; INDEX: fn-f.html

; PAGE: <title>f</title>
; PAGE: seq=0
; PAGE: data-loc="/tmp/show.c|8|3"
; PAGE: add i32 1, 2
