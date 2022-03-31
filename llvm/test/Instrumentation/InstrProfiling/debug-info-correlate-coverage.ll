; RUN: opt < %s -instrprof -debug-info-correlate -S | opt --O2 -S | FileCheck %s

@__profn_foo = private constant [3 x i8] c"foo"
; CHECK:      @__profc_foo

define  void @_Z3foov() !dbg !12 {
  call void @llvm.instrprof.cover(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @__profn_foo, i32 0, i32 0), i64 12345678, i32 1, i32 0)
  ; CHECK: store i8 0, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @__profc_foo
  ret void
}

declare void @llvm.instrprof.cover(i8*, i64, i32, i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-info-correlate-coverage.cpp", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 1, !"branch-target-enforcement", i32 0}
!6 = !{i32 1, !"sign-return-address", i32 0}
!7 = !{i32 1, !"sign-return-address-all", i32 0}
!8 = !{i32 1, !"sign-return-address-with-bkey", i32 0}
!9 = !{i32 7, !"uwtable", i32 1}
!10 = !{i32 7, !"frame-pointer", i32 1}
!11 = !{!"clang version 14.0.0"}
!12 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !13, file: !13, line: 1, type: !14, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
!13 = !DIFile(filename: "debug-info-correlate-coverage.cpp", directory: "")
!14 = !DISubroutineType(types: !15)
!15 = !{null}
!16 = !{}
