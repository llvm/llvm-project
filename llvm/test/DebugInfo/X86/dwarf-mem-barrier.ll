; Check .loc directive is emitted at correct position
; RUN: llc -O0 -mtriple x86_64-linux-gnu <%s | FileCheck %s

; CHECK:	#MEMBARRIER
; CHECK:	#MEMBARRIER
; CHECK:	movl	$0, -4(%rsp)
; CHECK:	.loc	1 1 14 prologue_end             # none.c:1:14

define dso_local i32 @main() !dbg !7 {
entry:
  fence acquire, !dbg !9
  fence release, !dbg !9
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  ret i32 0, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: DebugDirectivesOnly, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "none.c", directory: "/temp")
!2 = !{}
!3 = !{i32 2, !"Tracing", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 11.0.0"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 1, column: 14, scope: !7)
