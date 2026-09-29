;; Ensure that the loads, stores, and atomicrmw adds for gcov have nosanitize metadata.
; RUN: rm -rf %t && mkdir %t && cd %t
; RUN: opt < %s -S -passes=insert-gcov-profiling | FileCheck %s
; RUN: opt < %s -S -passes=insert-gcov-profiling -gcov-atomic-counter | FileCheck %s --check-prefixes=CHECK-ATOMIC-COUNTER

; CHECK-LABEL: void @empty()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %gcov_ctr = load i64, ptr @__llvm_gcov_ctr, align 4, !dbg [[DBG:![0-9]+]], !nosanitize [[NOSANITIZE:![0-9]+]]
; CHECK-NEXT:    %0 = add i64 %gcov_ctr, 1, !dbg [[DBG]]
; CHECK-NEXT:    store i64 %0, ptr @__llvm_gcov_ctr, align 4, !dbg [[DBG]], !nosanitize [[NOSANITIZE]]
; CHECK-NEXT:    ret void, !dbg [[DBG]]

; CHECK-ATOMIC-COUNTER-LABEL: void @empty()
; CHECK-ATOMIC-COUNTER-NEXT:  entry:
; CHECK-ATOMIC-COUNTER-NEXT:    %0 = atomicrmw add ptr @__llvm_gcov_ctr, i64 1 monotonic, align 8, !dbg [[DBG:![0-9]+]], !nosanitize [[NOSANITIZE:![0-9]+]]
; CHECK-ATOMIC-COUNTER-NEXT:    ret void, !dbg [[DBG]]

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
