; RUN: llvm-as -disable-output < %s -o /dev/null 2>&1 | FileCheck %s
; RUN: llvm-as < %s -o - | llvm-dis - | FileCheck %s --check-prefix=CHECK-STRIP
; CHECK: DICompileUnit not listed in llvm.dbg.cu
; CHECK: ignoring invalid debug info in
; CHECK-NOT: DICompileUnit not listed in llvm.dbg.cu
declare hidden void @g() local_unnamed_addr #1 align 2
define hidden void @f() {
  tail call void @g() #2, !llvm.loop !5
  ret void
}
!llvm.module.flags = !{!0, !1}
!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !{!5, !14, !15, !"fake loop metadata"}
; CHECK-STRIP: ![[MD:.*]] = distinct !{![[MD]], !{{[0-9]+}}, !{{[0-9]+}}, !"fake loop metadata"}
; CHECK-STRIP-NOT: DILocation
!7 = distinct !DISubprogram(name: "f", scope: !8, file: !8, line: 1324, type: !9, scopeLine: 1324, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !11)
!8 = !DIFile(filename: "/", directory: "f.cpp")
!9 = !DISubroutineType(types: !10)
!10 = !{}
!11 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !8)
!14 = !{!"metadata 1", i1 true}
!15 = !{!"metadata 1", !16}
!16 = distinct !{!16, !17}
!17 = !DILocation(line: 105, column: 3, scope: !18)
!18 = distinct !DILexicalBlock(scope: !7, file: !8, line: 105, column: 3)
