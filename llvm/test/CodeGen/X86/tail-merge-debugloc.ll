; RUN: llc -stop-after=branch-folder < %s | FileCheck %s
;
; bb2 and bb3 in the IR below will be tail-merged into a single basic block.
; As br instructions in bb2 and bb3 have the same debug location, make sure that
; the branch instruction in the merged basic block still maintains the debug 
; location info.
; 
; CHECK:      [[DLOC:![0-9]+]] = !DILocation(line: 2, column: 2, scope: !{{[0-9]+}})
; CHECK:      TEST64rr{{.*}}$rsi, renamable $rsi, implicit-def $eflags
; CHECK-NEXT: JCC_1{{.*}}, debug-location [[DLOC]]

target triple = "x86_64-unknown-linux-gnu"

define i32 @foo(i1 %b, i8* %p) {
bb1: 
  br i1 %b, label %bb2, label %bb3

bb2:
  %a1 = icmp eq i8* %p, null
  br i1 %a1, label %bb4, label %bb5, !dbg !6
  
bb3:
  %a2 = icmp eq i8* %p, null
  br i1 %a2, label %bb4, label %bb5, !dbg !6

bb4:
  ret i32 1

bb5:
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!1 = !DIFile(filename: "foo.c", directory: "b/")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0)
!5 = distinct !DILexicalBlock(scope: !4, file: !1, line: 1, column: 1)
!6 = !DILocation(line: 2, column: 2, scope: !5)
