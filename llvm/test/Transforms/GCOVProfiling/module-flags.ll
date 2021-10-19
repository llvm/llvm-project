; RUN: mkdir -p %t && cd %t
; RUN: opt < %s -S -passes=insert-gcov-profiling | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

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

;; Due to -fasynchronous-unwind-tables.
!9 = !{i32 7, !"uwtable", i32 1}

;; Due to -fno-omit-frame-pointer.
!10 = !{i32 7, !"frame-pointer", i32 2}

;; Infer uwtable and "frame-pointer" from the module flags.
; CHECK: define internal void @__llvm_gcov_writeout() unnamed_addr #[[#ATTR:]]
; CHECK: define internal void @__llvm_gcov_reset() unnamed_addr #[[#ATTR]]
; CHECK: define internal void @__llvm_gcov_init() unnamed_addr #[[#ATTR]]
; CHECK: attributes #[[#ATTR]] = { noinline nounwind uwtable "frame-pointer"="all" }
