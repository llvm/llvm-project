; RUN: opt -disable-output -passes=instcombine -ir-tracker-output=%t.tsv %s
; RUN: FileCheck %s --input-file=%t.tsv --check-prefix=ALL
; RUN: opt -disable-output -passes=instcombine -filter-print-funcs=f,h -ir-tracker-output=%t-cross.tsv %s
; RUN: FileCheck %s --input-file=%t-cross.tsv --check-prefix=CROSS
; RUN: opt -disable-output -passes=instcombine -filter-print-funcs=mid -ir-tracker-output=%t-mid.tsv %s
; RUN: FileCheck %s --input-file=%t-mid.tsv --check-prefix=MID
; RUN: opt -disable-output -passes=instcombine -filter-print-funcs=ssa -ir-tracker-output=%t-ssa.tsv %s
; RUN: FileCheck %s --input-file=%t-ssa.tsv --check-prefix=SSA
; RUN: opt -disable-output -passes=instcombine -filter-print-funcs=f -ir-tracker-output=%t-filter.tsv %s
; RUN: FileCheck %s --input-file=%t-filter.tsv --check-prefix=FILTER
; RUN: opt -disable-output -passes=instcombine -filter-print-funcs=flags -ir-tracker-output=%t-flags.tsv %s
; RUN: FileCheck %s --input-file=%t-flags.tsv --check-prefix=FLAGS

define i32 @f(i32 %x) !dbg !6 {
entry:
  %add = add i32 %x, 1, !dbg !8
  ret i32 %add, !dbg !9
}

define i32 @g(i32 %x) !dbg !7 {
entry:
  %mul = mul i32 %x, 2, !dbg !10
  ret i32 %mul, !dbg !11
}

define i32 @h(i32 %x) !dbg !12 {
entry:
  %add = add i32 %x, 1, !dbg !13
  ret i32 %add, !dbg !14
}

define i32 @mid(i32 %x) !dbg !16 {
entry:
  %a = freeze i32 %x, !dbg !17
  %b = mul i32 %a, 2, !dbg !18
  ret i32 %b, !dbg !19
}

define i32 @ssa(i32 %x) !dbg !20 {
entry:
  %a = add i32 %x, 1, !dbg !21
  %b = add i32 %a, 0, !dbg !22
  ret i32 %b, !dbg !25
}

define i32 @flags(i32 %x, i32 %y) !dbg !26 {
entry:
  %add = add nsw i32 %x, %y, !dbg !27
  %div = udiv exact i32 %add, 2, !dbg !28
  %cmp = icmp samesign slt i32 %div, %y, !dbg !29
  ret i32 %div, !dbg !30
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "ir-tracker-test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "ir-tracker.c", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!4 = !DISubroutineType(types: !5)
!5 = !{!3, !3}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 7, type: !4, scopeLine: 7, spFlags: DISPFlagDefinition, unit: !0)
!7 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 13, type: !4, scopeLine: 13, spFlags: DISPFlagDefinition, unit: !0)
!12 = distinct !DISubprogram(name: "h", scope: !15, file: !15, line: 7, type: !4, scopeLine: 7, spFlags: DISPFlagDefinition, unit: !0)
!16 = distinct !DISubprogram(name: "mid", scope: !1, file: !1, line: 19, type: !4, scopeLine: 19, spFlags: DISPFlagDefinition, unit: !0)
!20 = distinct !DISubprogram(name: "ssa", scope: !1, file: !1, line: 25, type: !4, scopeLine: 25, spFlags: DISPFlagDefinition, unit: !0)
!26 = distinct !DISubprogram(name: "flags", scope: !1, file: !1, line: 33, type: !4, scopeLine: 33, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 8, column: 3, scope: !6)
!9 = !DILocation(line: 9, column: 3, scope: !6)
!10 = !DILocation(line: 14, column: 3, scope: !7)
!11 = !DILocation(line: 15, column: 3, scope: !7)
!13 = !DILocation(line: 8, column: 3, scope: !12)
!14 = !DILocation(line: 9, column: 3, scope: !12)
!17 = !DILocation(line: 20, column: 3, scope: !16)
!18 = !DILocation(line: 21, column: 3, scope: !16)
!19 = !DILocation(line: 22, column: 3, scope: !16)
!21 = !DILocation(line: 26, column: 3, scope: !20)
!22 = !DILocation(line: 27, column: 3, scope: !20)
!25 = !DILocation(line: 30, column: 3, scope: !20)
!27 = !DILocation(line: 34, column: 3, scope: !26)
!28 = !DILocation(line: 35, column: 3, scope: !26)
!29 = !DILocation(line: 36, column: 3, scope: !26)
!30 = !DILocation(line: 37, column: 3, scope: !26)
!15 = !DIFile(filename: "ir-tracker-other.c", directory: "/tmp")

; Output is the cost-improvement TSV form: P/T/I rows. T rows carry source
; locations once per tracker ID; I rows reference tracker IDs.

; ALL: P{{	}}0{{	}}ir{{	}}initial{{	}}<initial>{{	}}f
; ALL-NEXT: T{{	}}{{[0-9]+}}{{	}}/tmp{{[/\\]}}ir-tracker.c{{	}}8{{	}}3
; ALL-NEXT: I{{	}}f{{	}}entry{{	}}0{{	}}add{{	}}{{[0-9]+}}{{	}}%add = add i32 %x, 1
; ALL-NEXT: T{{	}}{{[0-9]+}}{{	}}/tmp{{[/\\]}}ir-tracker.c{{	}}9{{	}}3
; ALL-NEXT: I{{	}}f{{	}}entry{{	}}1{{	}}ret{{	}}{{[0-9]+}}{{	}}ret i32 %add
; ALL: P{{	}}1{{	}}ir{{	}}after{{	}}instcombine{{	}}f
; ALL: P{{	}}2{{	}}ir{{	}}after{{	}}instcombine{{	}}g
; ALL-NEXT: T{{	}}{{[0-9]+}}{{	}}/tmp{{[/\\]}}ir-tracker.c{{	}}14{{	}}3
; ALL-NEXT: I{{	}}g{{	}}entry{{	}}0{{	}}shl{{	}}{{[0-9]+}}{{	}}%mul = shl i32 %x, 1
; ALL-NEXT: T{{	}}{{[0-9]+}}{{	}}/tmp{{[/\\]}}ir-tracker.c{{	}}15{{	}}3
; ALL-NEXT: I{{	}}g{{	}}entry{{	}}1{{	}}ret{{	}}{{[0-9]+}}{{	}}ret i32 %mul
; CROSS: P{{	}}0{{	}}ir{{	}}initial{{	}}<initial>{{	}}f
; CROSS-NEXT: T{{	}}{{[0-9]+}}{{	}}/tmp{{[/\\]}}ir-tracker.c{{	}}8{{	}}3
; CROSS-NEXT: I{{	}}f{{	}}entry{{	}}0{{	}}add{{	}}{{[0-9]+}}{{	}}%add = add i32 %x, 1
; CROSS-NEXT: T{{	}}{{[0-9]+}}{{	}}/tmp{{[/\\]}}ir-tracker.c{{	}}9{{	}}3
; CROSS-NEXT: I{{	}}f{{	}}entry{{	}}1{{	}}ret{{	}}{{[0-9]+}}{{	}}ret i32 %add
; CROSS: P{{	}}2{{	}}ir{{	}}after{{	}}instcombine{{	}}h
; CROSS-NEXT: T{{	}}{{[0-9]+}}{{	}}/tmp{{[/\\]}}ir-tracker-other.c{{	}}8{{	}}3
; CROSS-NEXT: I{{	}}h{{	}}entry{{	}}0{{	}}add{{	}}{{[0-9]+}}{{	}}%add = add i32 %x, 1
; CROSS-NEXT: T{{	}}{{[0-9]+}}{{	}}/tmp{{[/\\]}}ir-tracker-other.c{{	}}9{{	}}3
; CROSS-NEXT: I{{	}}h{{	}}entry{{	}}1{{	}}ret{{	}}{{[0-9]+}}{{	}}ret i32 %add

; MID: P{{	}}0{{	}}ir{{	}}initial{{	}}<initial>{{	}}mid
; MID-NEXT: T{{	}}{{[0-9]+}}{{	}}/tmp{{[/\\]}}ir-tracker.c{{	}}20{{	}}3
; MID-NEXT: I{{	}}mid{{	}}entry{{	}}0{{	}}freeze{{	}}{{[0-9]+}}{{	}}%a = freeze i32 %x
; MID-NEXT: T{{	}}{{[0-9]+}}{{	}}/tmp{{[/\\]}}ir-tracker.c{{	}}21{{	}}3
; MID-NEXT: I{{	}}mid{{	}}entry{{	}}1{{	}}mul{{	}}{{[0-9]+}}{{	}}%b = mul i32 %a, 2
; MID-NEXT: T{{	}}{{[0-9]+}}{{	}}/tmp{{[/\\]}}ir-tracker.c{{	}}22{{	}}3
; MID-NEXT: I{{	}}mid{{	}}entry{{	}}2{{	}}ret{{	}}{{[0-9]+}}{{	}}ret i32 %b
; MID: P{{	}}1{{	}}ir{{	}}after{{	}}instcombine{{	}}mid
; MID-NEXT: I{{	}}mid{{	}}entry{{	}}1{{	}}shl{{	}}{{[0-9]+}}{{	}}%b = shl i32 %a, 1

; SSA: P{{	}}0{{	}}ir{{	}}initial{{	}}<initial>{{	}}ssa
; SSA: I{{	}}ssa{{	}}entry{{	}}2{{	}}ret{{	}}{{[0-9]+}}{{	}}ret i32 %b
; SSA: P{{	}}1{{	}}ir{{	}}after{{	}}instcombine{{	}}ssa
; SSA-NEXT: I{{	}}ssa{{	}}entry{{	}}1{{	}}ret{{	}}{{[0-9]+}}{{	}}ret i32 %a

; FILTER: P{{	}}0{{	}}ir{{	}}initial{{	}}<initial>{{	}}f
; FILTER: I{{	}}f{{	}}entry{{	}}0{{	}}add
; FILTER: I{{	}}f{{	}}entry{{	}}1{{	}}ret
; FILTER-NOT: I{{	}}g{{	}}
; FILTER-NOT: ir_unit{{[" :]+}}g

; FLAGS: P{{	}}0{{	}}ir{{	}}initial{{	}}<initial>{{	}}flags
; FLAGS: I{{	}}flags{{	}}entry{{	}}0{{	}}add{{	}}{{[0-9]+}}{{	}}%add = add nsw i32 %x, %y
; FLAGS: I{{	}}flags{{	}}entry{{	}}1{{	}}udiv{{	}}{{[0-9]+}}{{	}}%div = udiv exact i32 %add, 2
; FLAGS: I{{	}}flags{{	}}entry{{	}}2{{	}}icmp{{	}}{{[0-9]+}}{{	}}%cmp = icmp samesign slt i1 %div, %y
