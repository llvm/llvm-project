; RUN: llc %s -enable-machine-outliner -mtriple=aarch64-unknown-unknown -pass-remarks=machine-outliner -pass-remarks-missed=machine-outliner -o /dev/null 2>&1 | FileCheck %s
; CHECK: <unknown>:0:0:
; CHECK-SAME: Did not outline 2 instructions from 2 locations.
; CHECK-SAME: Bytes from outlining all occurrences (16) >=
; CHECK-SAME: Unoutlined instruction bytes (16)
; CHECK-SAME: (Also found at: <UNKNOWN LOCATION>)
; CHECK: remark: <unknown>:0:0: Saved 36 bytes by outlining 11 instructions
; CHECK-SAME: from 2 locations. (Found at: <UNKNOWN LOCATION>,
; CHECK-SAME: <UNKNOWN LOCATION>)
; RUN: llc %s -enable-machine-outliner -mtriple=aarch64-unknown-unknown -o /dev/null -pass-remarks-missed=machine-outliner -pass-remarks-output=%t.yaml
; RUN: cat %t.yaml | FileCheck %s -check-prefix=YAML

; For the YAML case, the function we pick depends on the order of the candidate
; list.
; YAML: --- !Missed
; YAML-NEXT: Pass:            machine-outliner
; YAML-NEXT: Name:            NotOutliningCheaper
; YAML-NEXT: Function:
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Did not outline '
; YAML-NEXT:   - Length:          '2'
; YAML-NEXT:   - String:          ' instructions'
; YAML-NEXT:   - String:          ' from '
; YAML-NEXT:   - NumOccurrences:  '2'
; YAML-NEXT:   - String:          ' locations.'
; YAML-NEXT:   - String:          ' Bytes from outlining all occurrences ('
; YAML-NEXT:   - OutliningCost:   '16'
; YAML-NEXT:   - String:          ')'
; YAML-NEXT:   - String:          ' >= Unoutlined instruction bytes ('
; YAML-NEXT:   - NotOutliningCost: '16'
; YAML-NEXT:   - String:          ')'
; YAML-NEXT:   - String:          ' (Also found at: '
; YAML-NEXT:   - OtherStartLoc1:  '<UNKNOWN LOCATION>'
; YAML-NEXT:   - String:          ')'
; YAML: --- !Passed
; YAML-NEXT: Pass:            machine-outliner
; YAML-NEXT: Name:            OutlinedFunction
; YAML-NEXT: Function:        OUTLINED_FUNCTION_0
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          'Saved '
; YAML-NEXT:   - OutliningBenefit: '36'
; YAML-NEXT:   - String:          ' bytes by '
; YAML-NEXT:   - String:          'outlining '
; YAML-NEXT:   - Length:          '11'
; YAML-NEXT:   - String:          ' instructions '
; YAML-NEXT:   - String:          'from '
; YAML-NEXT:   - NumOccurrences:  '2'
; YAML-NEXT:   - String:          ' locations. '
; YAML-NEXT:   - String:          '(Found at: '
; YAML-NEXT:   - StartLoc0:       '<UNKNOWN LOCATION>'
; YAML-NEXT:   - String:          ', '
; YAML-NEXT:   - StartLoc1:       '<UNKNOWN LOCATION>'
; YAML-NEXT:   - String:          ')'

define void @dog() #0 !dbg !8 {
entry:
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  store i32 0, i32* %x, align 4
  store i32 1, i32* %y, align 4, !dbg !12
  ret void
}

define void @cat() #0 !dbg !14 {
entry:
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  store i32 0, i32* %x, align 4
  store i32 1, i32* %y, align 4, !dbg !16
  ret void
}

define void @foo() #0 !dbg !18 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 1, i32* %1, align 4, !dbg !24
  store i32 2, i32* %2, align 4
  store i32 3, i32* %3, align 4
  store i32 4, i32* %4, align 4
  store i32 5, i32* %5, align 4
  store i32 6, i32* %6, align 4, !dbg !26
  ret void
}

define void @bar() #0 !dbg !27 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 1, i32* %1, align 4, !dbg !33
  store i32 2, i32* %2, align 4
  store i32 3, i32* %3, align 4
  store i32 4, i32* %4, align 4
  store i32 5, i32* %5, align 4
  store i32 6, i32* %6, align 4, !dbg !35
  ret void
}

attributes #0 = { noredzone nounwind ssp uwtable "frame-pointer"="none" "target-cpu"="cyclone" }

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
!8 = distinct !DISubprogram(name: "dog", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!12 = !DILocation(line: 5, column: 9, scope: !8)
!14 = distinct !DISubprogram(name: "cat", scope: !1, file: !1, line: 10, type: !9, isLocal: false, isDefinition: true, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!16 = !DILocation(line: 13, column: 9, scope: !14)
!18 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 26, type: !9, isLocal: false, isDefinition: true, scopeLine: 26, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!24 = !DILocation(line: 27, column: 9, scope: !18)
!26 = !DILocation(line: 29, column: 9, scope: !18)
!27 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 35, type: !9, isLocal: false, isDefinition: true, scopeLine: 35, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!33 = !DILocation(line: 36, column: 1, scope: !27)
!35 = !DILocation(line: 38, column: 1, scope: !27)
