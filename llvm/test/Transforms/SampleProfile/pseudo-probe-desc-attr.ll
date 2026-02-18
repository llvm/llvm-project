
; REQUIRES: x86-registered-target
; RUN: opt < %s -passes=pseudo-probe -S -o - | FileCheck %s

; CHECK: ![[#]] = !{i64 {{-?[0-9]+}}, i64 [[#]], i8 0, !"baz"}
; CHECK: ![[#]] = !{i64 {{-?[0-9]+}}, i64 [[#]], i8 1, !"bar"}
; CHECK: ![[#]] = !{i64 {{-?[0-9]+}}, i64 [[#]], i8 2, !"foo"}

; Function Attrs: nounwind uwtable
define dso_local void @baz() #0 !dbg !11 {
entry:
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define dso_local void @bar() #1 !dbg !12 {
entry:
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local void @foo() #2 !dbg !13 {
entry:
  ret void
}

attributes #0 = { nounwind uwtable}
attributes #1 = { alwaysinline nounwind uwtable}
attributes #2 = { noinline nounwind uwtable}

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
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{}
!11 = distinct !DISubprogram(name: "baz", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !10)
!12 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 10, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !10)
!13 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 15, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !10)
