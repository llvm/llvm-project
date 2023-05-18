; RUN: llc %s -o - -generate-arange-section | FileCheck %s

; Make sure that emitting constants for ldr and emitting arange work together.
; Emitting constants must come before emitting aranges since emitting aranges can end arbitrary sections.

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-unknown-linux-android21"

; CHECK:       ldr r7, .Ltmp[[#TMP:]]

; CHECK:      .Ltmp[[#TMP]]:
; CHECK-NEXT: .long  83040

; CHECK: .section        .debug_aranges

define dso_local void @a() local_unnamed_addr !dbg !4 {
entry:
  call void asm sideeffect "  ldr r7, =${0:c}\0A", "i"(i32 83040)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/tmp/a.c", directory: "/tmp/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "a", scope: !5, file: !5, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!5 = !DIFile(filename: "/tmp/a.c", directory: "")
!6 = !DISubroutineType(types: !2)
