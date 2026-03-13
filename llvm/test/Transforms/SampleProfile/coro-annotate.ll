; RUN: opt < %s -sample-profile-file=%S/Inputs/coro-annotate.prof -passes=sample-profile -S | FileCheck %s

; This test checks if a coro function can correctly get sample annotation with the present of its await suspend wrappers.
; Please ignore the incomplete coro function body and missing coro intrinsics since only the function names matter here.
; Note: Do not change the function names. They are intentionally made to create a specific order in the module's ValSymTab iterator.

; CHECK: define void @bar() {{.*}} !prof ![[PROF:[0-9]+]]
; CHECK: ![[PROF]] = !{!"function_entry_count", i64 13294}
define void @bar() #0 !dbg !4 {
entry:
  ret void, !dbg !9
}

define internal void @bar.__await_suspend_wrapper__init(ptr noundef nonnull %0, ptr noundef %1) #1 {
entry:
  ret void
}

attributes #0 = { presplitcoroutine "use-sample-profile" }
attributes #1 = { "sample-profile-suffix-elision-policy"="selected" }

!llvm.module.flags = !{!7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{}
!4 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !2)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !DILocation(line: 1, column: 15, scope: !4)
