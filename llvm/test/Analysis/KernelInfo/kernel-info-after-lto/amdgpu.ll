; Check that -kernel-info-end-lto enables kernel-info in the AMD GPU target
; backend.

; REQUIRES: amdgpu-registered-target

; -kernel-info-end-lto inserts kernel-info into LTO pipeline.
; RUN: opt -pass-remarks=kernel-info -disable-output %s \
; RUN:     -passes='lto<O2>' -kernel-info-end-lto 2>&1 | \
; RUN:   FileCheck -match-full-lines %s

; Omitting -kernel-info-end-lto disables kernel-info.
; RUN: opt -pass-remarks=kernel-info -disable-output %s \
; RUN:     -passes='lto<O2>' 2>&1 | \
; RUN:   FileCheck -allow-empty -check-prefixes=NONE %s

; Omitting LTO disables kernel-info.
; RUN: opt -pass-remarks=kernel-info -disable-output %s \
; RUN:     -passes='default<O2>' -kernel-info-end-lto 2>&1 | \
; RUN:   FileCheck -allow-empty -check-prefixes=NONE %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; CHECK: remark: test.c:10:0: in artificial function 'test', OmpTargetNumTeams = 100
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
!nvvm.annotations = !{!6, !7, !8}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!2 = !DIFile(filename: "test.c", directory: "/tmp")
!3 = !{}
!4 = !DISubroutineType(types: !3)
!5 = distinct !DISubprogram(name: "test", scope: !2, file: !2, line: 10, type: !4, scopeLine: 10, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1, retainedNodes: !3)
!6 = !{ptr @test, !"maxclusterrank", i32 200}
!7 = !{ptr @test, !"maxntidx", i32 210}
!8 = distinct !{ptr null, !"kernel", i32 1}
