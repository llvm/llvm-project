; RUN: rm -f %t.tsv %t.db
; RUN: llc -enable-new-pm -mtriple=x86_64-unknown-linux-gnu -filetype=null -ir-tracker-output=%t.tsv %s
; RUN: FileCheck %s --input-file=%t.tsv --check-prefix=TSV
; RUN: %ir-tracker build --input %t.tsv --db %t.db
; RUN: %ir-tracker show --db %t.db --file mir-newpm.c --line 3 --kind mir --all-passes | FileCheck %s --check-prefix=SHOW

define i32 @f(i32 %x) !dbg !6 {
entry:
  %add = add i32 %x, 1, !dbg !8
  ret i32 %add, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "ir-tracker-test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "mir-newpm.c", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!4 = !DISubroutineType(types: !5)
!5 = !{!3, !3}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 3, column: 3, scope: !6)
!9 = !DILocation(line: 4, column: 3, scope: !6)

; TSV: P{{	}}{{[0-9]+}}{{	}}mir{{	}}initial{{	}}<initial>{{	}}f
; TSV: I{{	}}f{{	}}{{.*}}{{	}}{{[0-9]+}}{{	}}{{[^	]+}}{{	}}{{[0-9]+}}{{	}}{{.*}}
; TSV: P{{	}}{{[0-9]+}}{{	}}mir{{	}}after{{	}}{{[^	]+}}{{	}}f

; SHOW: seq={{[0-9]+}} [mir] '{{[^']+}}' on 'f'
; SHOW:   function f, block
