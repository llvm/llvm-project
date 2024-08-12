; Check info on launch bounds for AMD GPU.

; RUN: opt -pass-remarks=kernel-info -passes=kernel-info \
; RUN:     -disable-output %s 2>&1 | \
; RUN:   FileCheck -match-full-lines %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; CHECK: remark: test.c:10:0: in artificial function 'test', OmpTargetNumTeams = 100
; CHECK: remark: test.c:10:0: in artificial function 'test', OmpTargetThreadLimit = 101
; CHECK: remark: test.c:10:0: in artificial function 'test', AmdgpuMaxNumWorkgroupsX = 200
; CHECK: remark: test.c:10:0: in artificial function 'test', AmdgpuMaxNumWorkgroupsY = 201
; CHECK: remark: test.c:10:0: in artificial function 'test', AmdgpuMaxNumWorkgroupsZ = 202
; CHECK: remark: test.c:10:0: in artificial function 'test', AmdgpuFlatWorkGroupSizeMin = 210
; CHECK: remark: test.c:10:0: in artificial function 'test', AmdgpuFlatWorkGroupSizeMax = 211
; CHECK: remark: test.c:10:0: in artificial function 'test', AmdgpuWavesPerEuMin = 220
; CHECK: remark: test.c:10:0: in artificial function 'test', AmdgpuWavesPerEuMax = 221
define void @test() #0 !dbg !5 {
entry:
  ret void
}

attributes #0 = {
  "omp_target_num_teams"="100"
  "omp_target_thread_limit"="101"
  "amdgpu-max-num-workgroups"="200,201,202"
  "amdgpu-flat-work-group-size"="210,211"
  "amdgpu-waves-per-eu"="220,221"
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!2 = !DIFile(filename: "test.c", directory: "/tmp")
!3 = !{}
!4 = !DISubroutineType(types: !3)
!5 = distinct !DISubprogram(name: "test", scope: !2, file: !2, line: 10, type: !4, scopeLine: 10, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1, retainedNodes: !3)
