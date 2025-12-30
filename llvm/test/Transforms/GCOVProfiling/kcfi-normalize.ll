;; Ensure __llvm_gcov_(writeout|reset|init) have the correct !kcfi_type
;; with integer normalization.
; RUN: mkdir -p %t && cd %t
; RUN: opt < %s -S -passes=insert-gcov-profiling \
; RUN:  -mtriple=x86_64-unknown-linux-gnu | FileCheck \
; RUN:  --check-prefixes=CHECK,CHECK-CTOR-INIT %s
; RUN: opt < %s -S -passes=insert-gcov-profiling \
; RUN:  -mtriple=powerpc64-ibm-aix | FileCheck \
; RUN:  --check-prefixes=CHECK,CHECK-RT-INIT %s

; Check for gcov initialization function pointers when we initialize
; the writeout and reset functions in the runtime.
; CHECK-RT-INIT: @__llvm_covinit_functions = private constant { ptr, ptr } { ptr @__llvm_gcov_writeout, ptr @__llvm_gcov_reset }, section "__llvm_covinit"

define dso_local void @empty() !dbg !5 {
entry:
  ret void, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "a.c", directory: "")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "empty", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !DILocation(line: 2, column: 1, scope: !5)
!9 = !{i32 4, !"kcfi", i32 1}
!10 = !{i32 4, !"cfi-normalize-integers", i32 1}

; CHECK: define internal void @__llvm_gcov_writeout()
; CHECK-SAME: !kcfi_type ![[#TYPE:]]
; CHECK: define internal void @__llvm_gcov_reset()
; CHECK-SAME: !kcfi_type ![[#TYPE]]
; CHECK-CTOR-INIT: define internal void @__llvm_gcov_init()
; CHECK-CTOR-INIT-SAME: !kcfi_type ![[#TYPE]]

; CHECK: ![[#TYPE]] = !{i32 -440107680}
