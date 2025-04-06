; Check info on launch bounds for AMD GPU.

; REQUIRES: amdgpu-registered-target

; RUN: opt -pass-remarks=kernel-info -passes=kernel-info \
; RUN:     -disable-output %s 2>&1 | \
; RUN:   FileCheck -match-full-lines %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; CHECK: remark: test.c:10:0: in artificial function 'all', omp_target_num_teams = 100
; CHECK: remark: test.c:10:0: in artificial function 'all', omp_target_thread_limit = 101
; CHECK: remark: test.c:10:0: in artificial function 'all', amdgpu-max-num-workgroups[0] = 200
; CHECK: remark: test.c:10:0: in artificial function 'all', amdgpu-max-num-workgroups[1] = 201
; CHECK: remark: test.c:10:0: in artificial function 'all', amdgpu-max-num-workgroups[2] = 202
; CHECK: remark: test.c:10:0: in artificial function 'all', amdgpu-flat-work-group-size[0] = 210
; CHECK: remark: test.c:10:0: in artificial function 'all', amdgpu-flat-work-group-size[1] = 211
; CHECK: remark: test.c:10:0: in artificial function 'all', amdgpu-waves-per-eu[0] = 2
; CHECK: remark: test.c:10:0: in artificial function 'all', amdgpu-waves-per-eu[1] = 9
define void @all() #0 !dbg !5 {
entry:
  ret void
}

; CHECK-NOT: remark: test.c:11:0: in function 'none', omp_target_num_teams = {{.*}}
; CHECK-NOT: remark: test.c:11:0: in function 'none', omp_target_thread_limit = {{.*}}
; CHECK: remark: test.c:11:0: in function 'none', amdgpu-max-num-workgroups[0] = 4294967295
; CHECK: remark: test.c:11:0: in function 'none', amdgpu-max-num-workgroups[1] = 4294967295
; CHECK: remark: test.c:11:0: in function 'none', amdgpu-max-num-workgroups[2] = 4294967295
; CHECK: remark: test.c:11:0: in function 'none', amdgpu-flat-work-group-size[0] = 1
; CHECK: remark: test.c:11:0: in function 'none', amdgpu-flat-work-group-size[1] = 1024
; CHECK: remark: test.c:11:0: in function 'none', amdgpu-waves-per-eu[0] = 4
; CHECK: remark: test.c:11:0: in function 'none', amdgpu-waves-per-eu[1] = 10
define void @none() !dbg !6 {
entry:
  ret void
}

; CHECK: remark: test.c:12:0: in function 'bogus', omp_target_num_teams = 987654321
; CHECK: remark: test.c:12:0: in function 'bogus', omp_target_thread_limit = 987654321
; CHECK: remark: test.c:12:0: in function 'bogus', amdgpu-max-num-workgroups[0] = 987654321
; CHECK: remark: test.c:12:0: in function 'bogus', amdgpu-max-num-workgroups[1] = 987654321
; CHECK: remark: test.c:12:0: in function 'bogus', amdgpu-max-num-workgroups[2] = 987654321
; CHECK: remark: test.c:12:0: in function 'bogus', amdgpu-flat-work-group-size[0] = 1
; CHECK: remark: test.c:12:0: in function 'bogus', amdgpu-flat-work-group-size[1] = 1024
; CHECK: remark: test.c:12:0: in function 'bogus', amdgpu-waves-per-eu[0] = 4
; CHECK: remark: test.c:12:0: in function 'bogus', amdgpu-waves-per-eu[1] = 10
define void @bogus() #1 !dbg !7 {
entry:
  ret void
}

attributes #0 = {
  "omp_target_num_teams"="100"
  "omp_target_thread_limit"="101"
  "amdgpu-max-num-workgroups"="200,201,202"
  "amdgpu-flat-work-group-size"="210,211"
  "amdgpu-waves-per-eu"="2,9"
}

; We choose values that are small enough to parse successfully but that are
; impossibly large.  For values that are validated, we check that they are
; overridden with realistic values.
attributes #1 = {
  "omp_target_num_teams"="987654321"
  "omp_target_thread_limit"="987654321"
  "amdgpu-max-num-workgroups"="987654321,987654321,987654321"
  "amdgpu-flat-work-group-size"="987654321,987654321"
  "amdgpu-waves-per-eu"="987654321,987654321"
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!2 = !DIFile(filename: "test.c", directory: "/tmp")
!3 = !{}
!4 = !DISubroutineType(types: !3)
!5 = distinct !DISubprogram(name: "all", scope: !2, file: !2, line: 10, type: !4, scopeLine: 10, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1, retainedNodes: !3)
!6 = distinct !DISubprogram(name: "none", scope: !2, file: !2, line: 11, type: !4, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1, retainedNodes: !3)
!7 = distinct !DISubprogram(name: "bogus", scope: !2, file: !2, line: 12, type: !4, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1, retainedNodes: !3)
