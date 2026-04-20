; RUN: llc -mtriple=i386-windows-msvc -filetype=asm < %s | FileCheck %s

; CHECK: #APP
; CHECK: {{[ \t]*}}.cv_loc{{[ \t]+}}0 1 6 9
; CHECK-NEXT: {{[ \t]*}}nop
; CHECK-NEXT: {{[ \t]*}}.cv_loc{{[ \t]+}}0 1 7 9
; CHECK-NEXT: {{[ \t]*}}nop
; CHECK: #NO_APP

target triple = "i386-pc-windows-msvc"

define dso_local i32 @main() !dbg !8 {
entry:
  ; The inline asm string is two source instructions: "nop\n\tnop".
  call void asm sideeffect inteldialect "nop\0A\09nop", "~{dirflag},~{fpsr},~{flags}"(), !srcloc !12, !dbg !13
  ret i32 0, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.cpp", directory: "/")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 3, type: !5, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!12 = !{i64 0, i64 0, !14}
!13 = !DILocation(line: 4, column: 5, scope: !8)
!14 = !{!"inlineasm.dbg.line", i32 6, i32 9, i32 7, i32 9}
!15 = !DILocation(line: 8, column: 3, scope: !8)
