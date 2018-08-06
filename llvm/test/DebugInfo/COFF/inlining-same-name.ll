; RUN: llc -mtriple=x86_64-windows-msvc < %s -filetype=obj -o - | llvm-readobj - -codeview | FileCheck %s

; We should only get one func id record, and both inlinees should point to it,
; even though there are two DISubprograms.

; CHECK:  FuncId (0x1002) {
; CHECK-NEXT:    TypeLeafKind: LF_FUNC_ID (0x1601)
; CHECK-NEXT:    ParentScope: 0x0
; CHECK-NEXT:    FunctionType: void () (0x1001)
; CHECK-NEXT:    Name: same_name
; CHECK-NEXT:  }
; CHECK-NOT:    Name: same_name

; CHECK: CodeViewDebugInfo [
; CHECK:   Section: .debug$S
; CHECK:   Subsection [
; CHECK:     {{.*}}Proc{{.*}}Sym {
; CHECK:       DisplayName: main
; CHECK:     }
; CHECK:     InlineSiteSym {
; CHECK:       Inlinee: same_name (0x1002)
; CHECK:     }
; CHECK:     InlineSiteEnd {
; CHECK:     }
; CHECK:     InlineSiteSym {
; CHECK:       Inlinee: same_name (0x1002)
; CHECK:     }
; CHECK:     InlineSiteEnd {
; CHECK:     }
; CHECK:     ProcEnd
; CHECK:   ]

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @main(i32* %i.i) !dbg !16 {
  store volatile i32 3, i32* %i.i, !dbg !6
  store volatile i32 3, i32* %i.i, !dbg !19
  ret void
}

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!4}

!0 = !{i32 2, !"CodeView", i32 1}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DICompileUnit(language: DW_LANG_D, file: !5, producer: "LDC (http://wiki.dlang.org/LDC)", isOptimized: false, runtimeVersion: 1, emissionKind: FullDebug)
!5 = !DIFile(filename: "opover2.d", directory: "C:\5CLDC\5Cninja-ldc\5C..\5Cldc\5Ctests\5Cd2\5Cdmd-testsuite\5Crunnable")
!6 = !DILocation(line: 302, column: 9, scope: !7, inlinedAt: !15)
!7 = distinct !DISubprogram(name: "same_name", linkageName: "same_name", scope: null, file: !5, line: 302, type: !8, isLocal: false, isDefinition: true, scopeLine: 302, flags: DIFlagPrototyped, isOptimized: false, unit: !4, retainedNodes: !{})
!8 = !DISubroutineType(types: !{})
!15 = distinct !DILocation(line: 333, column: 5, scope: !16)
!16 = distinct !DISubprogram(name: "main", linkageName: "main", scope: null, file: !5, line: 328, type: !8, isLocal: false, isDefinition: true, scopeLine: 328, flags: DIFlagPrototyped, isOptimized: false, unit: !4, retainedNodes: !{})
!19 = !DILocation(line: 308, column: 9, scope: !20, inlinedAt: !25)
!20 = distinct !DISubprogram(name: "same_name", linkageName: "same_name",  scope: null, file: !5, line: 308, type: !8, isLocal: false, isDefinition: true, scopeLine: 308, flags: DIFlagPrototyped, isOptimized: false, unit: !4, retainedNodes: !{})
!25 = distinct !DILocation(line: 334, column: 5, scope: !16)
