; RUN: mkdir -p %t && cd %t
; RUN: opt < %s -S -passes=insert-gcov-profiling | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

define void @foo() !dbg !8 {
entry:
  ret void, !dbg !12
}

define void @bar() !dbg !13 {
entry:
  ret void, !dbg !14
}

; CHECK: define internal void @__llvm_gcov_reset()
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @llvm.memset.p0.i64(ptr @__llvm_gcov_ctr, i8 0, i64 8, i1 false)
; CHECK-NEXT: call void @llvm.memset.p0.i64(ptr @__llvm_gcov_ctr.1, i8 0, i64 8, i1 false)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/tmp/a.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"uwtable", i32 1}
!8 = distinct !DISubprogram(name: "foo", scope: !9, file: !9, line: 1, type: !10, scopeLine: 1, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!9 = !DIFile(filename: "/tmp/a.c", directory: "")
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !DILocation(line: 1, column: 13, scope: !8)
!13 = distinct !DISubprogram(name: "bar", scope: !9, file: !9, line: 2, type: !10, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!14 = !DILocation(line: 2, column: 13, scope: !13)
