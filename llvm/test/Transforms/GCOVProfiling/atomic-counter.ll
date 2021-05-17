;; -fsanitize=thread requires the (potentially concurrent) counter updates to be atomic.
; RUN: mkdir -p %t && cd %t
; RUN: opt < %s -S -passes=insert-gcov-profiling -gcov-atomic-counter | FileCheck %s

; CHECK-LABEL: void @empty()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %0 = atomicrmw add i64* getelementptr inbounds ([1 x i64], [1 x i64]* @__llvm_gcov_ctr, i64 0, i64 0), i64 1 monotonic, align 8, !dbg [[DBG:![0-9]+]]
; CHECK-NEXT:    ret void, !dbg [[DBG]]

define dso_local void @empty() !dbg !5 {
entry:
  ret void, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "a.c", directory: "")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "empty", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !DILocation(line: 2, column: 1, scope: !5)
