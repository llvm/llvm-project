;; Ensure no relocations are emitted into .debug_*.dwo sections.
;; This used to fail when relaxation was enabled.

; RUN: llc --mtriple=loongarch64-unknown-linux-gnu --dwarf-version=5 \
; RUN:     --split-dwarf-file=foo.dwo --split-dwarf-output=%t.dwo --mattr=+relax --filetype=obj %s -o %t.o
; RUN: llvm-readobj -r %t.dwo | FileCheck %s --check-prefix=DWO-RELOC

; DWO-RELOC: Relocations [
; DWO-RELOC-NEXT: ]

@.str = external constant [3 x i8]

define i64 @_ZN15partition_alloc8internal4base7strings8internal12SafeSNPrintfEPcmPKcPKNS3_3ArgEm(i1 %cmp14.i622) !dbg !3 {
entry:
  %0 = load i8, ptr @.str, align 1
  br i1 %cmp14.i622, label %while.body.i.preheader, label %for.inc.preheader.i, !dbg !6

while.body.i.preheader:                           ; preds = %entry
  store i8 %0, ptr null, align 1
  ret i64 0

for.inc.preheader.i:                              ; preds = %entry
  %strlen.i = load volatile i64, ptr @.str, align 8
  br label %do.body61.i

do.body61.i:                                      ; preds = %do.body61.i, %for.inc.preheader.i
  %reverse_prefix.2.i = phi ptr [ null, %for.inc.preheader.i ], [ %incdec.ptr98.i, %do.body61.i ]
  %cmp96.i = icmp ugt ptr %reverse_prefix.2.i, null
  %incdec.ptr98.i = getelementptr i8, ptr %reverse_prefix.2.i, i64 -1, !dbg !9
  br label %do.body61.i
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 22.1.2 (https://github.com/llvm/llvm-project.git 6121df77a781d2e6f9a8e569aa45ccfffd6c7e0e)", isOptimized: true, runtimeVersion: 0, splitDebugFilename: "safe_sprintf.dwo", emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: GNU)
!1 = !DIFile(filename: "../../third_party/partition_alloc/src/partition_alloc/partition_alloc_base/strings/safe_sprintf.cc", directory: "/home/wanglei")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "SafeSNPrintf", scope: !1, file: !1, line: 437, type: !4, scopeLine: 441, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !DILocation(line: 326, column: 13, scope: !7, inlinedAt: !8, atomGroup: 23, atomRank: 1)
!7 = distinct !DISubprogram(name: "IToASCII", scope: !1, file: !1, line: 276, type: !4, scopeLine: 282, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!8 = distinct !DILocation(line: 597, column: 18, scope: !3)
!9 = !DILocation(line: 385, column: 15, scope: !7, inlinedAt: !8, atomGroup: 57, atomRank: 2)
