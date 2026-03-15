; CHECK: remark: test.c:10:0: in artificial function 'test', omp_target_num_teams = 100
; NONE-NOT: remark:
define void @test() #0 !dbg !5 {
entry:
  ret void
}

attributes #0 = {
  "omp_target_num_teams"="100"
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}
!nvvm.annotations = !{!6}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!2 = !DIFile(filename: "test.c", directory: "/tmp")
!3 = !{}
!4 = !DISubroutineType(types: !3)
!5 = distinct !DISubprogram(name: "test", scope: !2, file: !2, line: 10, type: !4, scopeLine: 10, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1, retainedNodes: !3)
!6 = distinct !{ptr null, !"kernel", i32 1}
