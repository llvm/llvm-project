; RUN: llc -mtriple=i386-windows-msvc -x86-asm-syntax=intel -filetype=asm < %s | FileCheck %s

; CHECK: #APP
; CHECK: {{[ \t]*}}.cv_loc{{[ \t]+}}0 1 6 9
; CHECK-NEXT: {{[ \t]*}}lea eax, [a]
; CHECK-NEXT: {{[ \t]*}}.cv_loc{{[ \t]+}}0 1 7 9
; CHECK-NEXT: {{[ \t]*}}mov dword ptr [eax], 1
; CHECK: {{[ \t]*}}.cv_loc{{[ \t]+}}0 1 9 9
; CHECK-NEXT: {{[ \t]*}}lea ebx, [b]
; CHECK: {{[ \t]*}}.cv_loc{{[ \t]+}}0 1 10 9
; CHECK-NEXT: {{[ \t]*}}mov dword ptr [ebx], 1
; CHECK: {{[ \t]*}}.cv_loc{{[ \t]+}}0 1 12 9
; CHECK-NEXT: {{[ \t]*}}mov eax, dword ptr [eax]
; CHECK: {{[ \t]*}}.cv_loc{{[ \t]+}}0 1 13 9
; CHECK-NEXT: {{[ \t]*}}add dword ptr [ebx], eax
; CHECK: {{[ \t]*}}.cv_loc{{[ \t]+}}0 1 15 9
; CHECK-NEXT: {{[ \t]*}}inc eax
; CHECK: {{[ \t]*}}.cv_loc{{[ \t]+}}0 1 17 9
; CHECK-NEXT: {{[ \t]*}}imul dword ptr [ebx]
; CHECK: {{[ \t]*}}.cv_loc{{[ \t]+}}0 1 18 9
; CHECK-NEXT: {{[ \t]*}}mov dword ptr [ebx], eax
; CHECK: #NO_APP

target triple = "i386-pc-windows-msvc"

@a = dso_local global i32 0
@b = dso_local global i32 0

define dso_local i32 @main() !dbg !8 {
entry:
  ; The inline asm string contains multi-byte instructions and blank lines.
  call void asm sideeffect inteldialect "lea eax, a\0A\09mov dword ptr [eax], 1\0A\0A\09lea ebx, b\0A\09mov dword ptr [ebx], 1\0A\0A\09mov eax, [eax]\0A\09add [ebx], eax\0A\0A\09inc eax\0A\0A\09imul dword ptr [ebx]\0A\09mov [ebx], eax", "~{dirflag},~{fpsr},~{flags}"(), !srcloc !12, !dbg !13
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
!12 = !{i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0, !14}
!13 = !DILocation(line: 4, column: 5, scope: !8)
!14 = !{!"inlineasm.dbg.offset", i32 0, i32 6, i32 9, i32 12, i32 7, i32 9, i32 37, i32 9, i32 9, i32 49, i32 10, i32 9, i32 74, i32 12, i32 9, i32 90, i32 13, i32 9, i32 107, i32 15, i32 9, i32 117, i32 17, i32 9, i32 139, i32 18, i32 9}
!15 = !DILocation(line: 8, column: 3, scope: !8)
