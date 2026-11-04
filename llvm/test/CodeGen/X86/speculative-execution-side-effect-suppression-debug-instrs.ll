; RUN: llc -mtriple=i686-unknown-linux-gnu -verify-machineinstrs %s -o - | FileCheck %s

; A DBG_VALUE after an LFENCE should not make SESES insert another LFENCE
; before the following load.

@probe = external global i32

define i32 @read_after_fence(i32 %x) nounwind "target-features"="+sse2,+seses" !dbg !4 {
; CHECK-LABEL: read_after_fence:
; CHECK:         lfence
; CHECK:         #DEBUG_VALUE: read_after_fence:x <- undef
; CHECK-NOT:     lfence
; CHECK:         movl probe, %eax
entry:
  call void @llvm.x86.sse2.lfence(), !dbg !8
    #dbg_value(i32 %x, !7, !DIExpression(), !8)
  %value = load volatile i32, ptr @probe, align 4, !dbg !8
  %sum = add i32 %value, %x, !dbg !8
  ret i32 %sum, !dbg !8
}

declare void @llvm.x86.sse2.lfence()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "llvm", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "seses-debug-instr.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "read_after_fence", linkageName: "read_after_fence", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!5 = !DISubroutineType(types: !10)
!6 = !{!7}
!7 = !DILocalVariable(name: "x", scope: !4, file: !1, line: 1, type: !9)
!8 = !DILocation(line: 1, column: 1, scope: !4)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{!9, !9}
