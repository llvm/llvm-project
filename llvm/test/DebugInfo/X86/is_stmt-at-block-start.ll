;; Checks that when an instruction at the start of a BasicBlock has the same
;; DebugLoc as the instruction at the end of the previous BasicBlock, we add
;; is_stmt to the new line, to ensure that we still step on it if we arrive from
;; a BasicBlock other than the immediately preceding one.

; RUN: %llc_dwarf -mtriple=x86_64-unknown-linux -O0 -filetype=obj < %s | llvm-dwarfdump --debug-line - | FileCheck %s

; CHECK:      {{0x[0-9a-f]+}}     13      5 {{.+}} is_stmt
; CHECK-NEXT: {{0x[0-9a-f]+}}     13      5 {{.+}} is_stmt

define void @_Z1fi(i1 %cond) !dbg !21 {
entry:
  br i1 %cond, label %if.then2, label %if.else4

if.then2:                                         ; preds = %entry
  br label %if.end8, !dbg !24

if.else4:                                         ; preds = %entry
  %0 = load i32, ptr null, align 4, !dbg !24
  %call5 = call i1 null(i32 %0)
  ret void

if.end8:                                          ; preds = %if.then2
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 20.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/home/gbtozers/dev/upstream-llvm")
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = distinct !DISubprogram(name: "f", linkageName: "_Z1fi", scope: !1, file: !1, line: 7, type: !22, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!22 = distinct !DISubroutineType(types: !23)
!23 = !{null}
!24 = !DILocation(line: 13, column: 5, scope: !25)
!25 = distinct !DILexicalBlock(scope: !21, file: !1, line: 11, column: 27)
