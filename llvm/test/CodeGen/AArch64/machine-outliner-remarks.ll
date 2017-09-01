; RUN: llc %s -enable-machine-outliner -mtriple=aarch64-unknown-unknown -pass-remarks-missed=machine-outliner -o /dev/null 2>&1 | FileCheck %s
; CHECK: machine-outliner-remarks.ll:5:9:
; CHECK-SAME: Did not outline 2 instructions from 2 locations.
; CHECK-SAME: Instructions from outlining all occurrences (9) >=
; CHECK-SAME: Unoutlined instruction count (4)
; CHECK-SAME: (Also found at: machine-outliner-remarks.ll:13:9)
; RUN: llc %s -enable-machine-outliner -mtriple=aarch64-unknown-unknown -o /dev/null -pass-remarks-missed=machine-outliner -pass-remarks-output=%t.yaml
; RUN: cat %t.yaml | FileCheck %s -check-prefix=YAML
; YAML: --- !Missed
; YAML-NEXT: Pass:            machine-outliner
; YAML-NEXT: Name:            NotOutliningCheaper
; YAML-NEXT: DebugLoc:        { File: machine-outliner-remarks.ll, Line: 5, Column: 9 }
; YAML-NEXT: Function:        dog
; YAML-NEXT: Args:            
; YAML-NEXT:   - String:          'Did not outline '
; YAML-NEXT:   - Length:          '2'
; YAML-NEXT:   - String:          ' instructions'
; YAML-NEXT:   - String:          ' from '
; YAML-NEXT:   - NumOccurrences:  '2'
; YAML-NEXT:   - String:          ' locations.'
; YAML-NEXT:   - String:          ' Instructions from outlining all occurrences ('
; YAML-NEXT:   - OutliningCost:   '9'
; YAML-NEXT:   - String:          ')'
; YAML-NEXT:   - String:          ' >= Unoutlined instruction count ('
; YAML-NEXT:   - NotOutliningCost: '4'
; YAML-NEXT:   - String:          ')'
; YAML-NEXT:   - String:          ' (Also found at: '
; YAML-NEXT:   - OtherStartLoc1:  'machine-outliner-remarks.ll:13:9'
; YAML-NEXT:     DebugLoc:        { File: machine-outliner-remarks.ll, Line: 13, Column: 9 }
; YAML-NEXT:   - String:          ')'

define void @dog() #0 !dbg !8 {
entry:
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  store i32 0, i32* %x, align 4, !dbg !11
  store i32 1, i32* %y, align 4, !dbg !12
  ret void, !dbg !13
}

define void @cat() #0 !dbg !14 {
entry:
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  store i32 0, i32* %x, align 4, !dbg !15
  store i32 1, i32* %y, align 4, !dbg !16
  ret void, !dbg !17
}

attributes #0 = { noredzone nounwind ssp uwtable "no-frame-pointer-elim"="false" "target-cpu"="cyclone" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "machine-outliner-remarks.ll", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!""}
!8 = distinct !DISubprogram(name: "dog", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocation(line: 4, column: 9, scope: !8)
!12 = !DILocation(line: 5, column: 9, scope: !8)
!13 = !DILocation(line: 6, column: 1, scope: !8)
!14 = distinct !DISubprogram(name: "cat", scope: !1, file: !1, line: 10, type: !9, isLocal: false, isDefinition: true, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!15 = !DILocation(line: 12, column: 9, scope: !14)
!16 = !DILocation(line: 13, column: 9, scope: !14)
!17 = !DILocation(line: 14, column: 1, scope: !14)
