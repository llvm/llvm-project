; REQUIRES: x86-registered-target
; RUN: opt < %s -passes=pseudo-probe -S -o - | FileCheck %s

; CHECK: ![[#]] = !{i64 -3345296970173352005, i64 [[#]], !"foo.dbg"}
; CHECK-NOT: ![[#]] = !{i64 6699318081062747564, i64 [[#]], !"foo"}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo() #0 !dbg !8 {
entry:
  ret void, !dbg !12
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{!"clang version 17.0.0 "}
!8 = distinct !DISubprogram(name: "foo.dbg", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !11)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !{}
!12 = !DILocation(line: 1, column: 13, scope: !8)
