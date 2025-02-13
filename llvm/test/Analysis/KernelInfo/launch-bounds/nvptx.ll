; Check info on launch bounds for NVPTX.

; REQUIRES: nvptx-registered-target

; RUN: opt -pass-remarks=kernel-info -passes=kernel-info \
; RUN:     -disable-output %s 2>&1 | \
; RUN:   FileCheck -match-full-lines %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; CHECK: remark: test.c:10:0: in artificial function 'test', omp_target_num_teams = 100
; CHECK: remark: test.c:10:0: in artificial function 'test', omp_target_thread_limit = 101
; CHECK: remark: test.c:10:0: in artificial function 'test', maxclusterrank = 200
; CHECK: remark: test.c:10:0: in artificial function 'test', maxntidx = 210
; CHECK: remark: test.c:10:0: in artificial function 'test', maxntidy = 211
; CHECK: remark: test.c:10:0: in artificial function 'test', maxntidz = 212
define void @test() #0 !dbg !5 {
entry:
  ret void
}

attributes #0 = {
  "omp_target_num_teams"="100"
  "omp_target_thread_limit"="101"
  "nvvm.maxclusterrank"="200"
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}
!nvvm.annotations = !{!7, !8, !9, !10}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!2 = !DIFile(filename: "test.c", directory: "/tmp")
!3 = !{}
!4 = !DISubroutineType(types: !3)
!5 = distinct !DISubprogram(name: "test", scope: !2, file: !2, line: 10, type: !4, scopeLine: 10, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1, retainedNodes: !3)
!7 = !{ptr @test, !"maxntidx", i32 210}
!8 = !{ptr @test, !"maxntidy", i32 211}
!9 = !{ptr @test, !"maxntidz", i32 212}
!10 = distinct !{ptr null, !"kernel", i32 1}
