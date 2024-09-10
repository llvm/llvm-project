; RUN: opt -S %s -passes=verify 2>&1 \
; RUN: | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -S %s -passes=verify 2>&1 \
; RUN: | FileCheck %s

;; NOTE: Expect opt to return zero because the badly formed debug info
;; is going to be stripped.

;; Check that badly formed assignment tracking metadata is caught either
;; while parsing or by the verifier.

;; Check verifier output.
; CHECK: !DIAssignID attached to unexpected instruction kind

;; Check DIAssignID is stripped from IR.
; CHECK: define dso_local void @fun() {
; CHECK-NOT: DIAssignID

define dso_local void @fun() !dbg !7 {
entry:
  ret void, !DIAssignID !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !1000}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 14.0.0"}
!7 = distinct !DISubprogram(name: "fun", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!14 = distinct !DIAssignID()
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
