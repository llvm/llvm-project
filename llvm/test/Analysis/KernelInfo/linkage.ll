; Check info on linkage.

; RUN: opt -pass-remarks=kernel-info -passes=kernel-info \
; RUN:     -disable-output %s 2>&1 | \
; RUN:   FileCheck -match-full-lines %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; CHECK: remark: test.c:13:0: in artificial function 'extNotKer', ExternalNotKernel = 1
define external void @extNotKer() !dbg !10 {
entry:
  ret void
}

; CHECK: remark: test.c:23:0: in function 'impNotKer', ExternalNotKernel = 1
define void @impNotKer() !dbg !20 {
entry:
  ret void
}

; CHECK: remark: test.c:33:0: in artificial function 'weakNotKer', ExternalNotKernel = 0
define weak void @weakNotKer() !dbg !30 {
entry:
  ret void
}

; CHECK: remark: test.c:43:0: in function 'extPtxKer', ExternalNotKernel = 0
define external ptx_kernel void @extPtxKer() !dbg !40 {
entry:
  ret void
}

; CHECK: remark: test.c:53:0: in artificial function 'extAmdgpuKer', ExternalNotKernel = 0
define external amdgpu_kernel void @extAmdgpuKer() !dbg !50 {
entry:
  ret void
}

; CHECK: remark: test.c:63:0: in function 'extSpirKer', ExternalNotKernel = 0
define external spir_kernel void @extSpirKer() !dbg !60 {
entry:
  ret void
}

; CHECK: remark: test.c:73:0: in artificial function 'weakKer', ExternalNotKernel = 0
define weak ptx_kernel void @weakKer() !dbg !70 {
entry:
  ret void
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!2 = !DIFile(filename: "test.c", directory: "/tmp")
!3 = !{null}
!4 = !{}
!5 = !DISubroutineType(types: !3)

!10 = distinct !DISubprogram(name: "extNotKer", scope: !2, file: !2, line: 13, type: !5, scopeLine: 13, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !4)
!20 = distinct !DISubprogram(name: "impNotKer", scope: !2, file: !2, line: 23, type: !5, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !4)
!30 = distinct !DISubprogram(name: "weakNotKer", scope: !2, file: !2, line: 33, type: !5, scopeLine: 33, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !4)
!40 = distinct !DISubprogram(name: "extPtxKer", scope: !2, file: !2, line: 43, type: !5, scopeLine: 43, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !4)
!50 = distinct !DISubprogram(name: "extAmdgpuKer", scope: !2, file: !2, line: 53, type: !5, scopeLine: 53, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !4)
!60 = distinct !DISubprogram(name: "extSpirKer", scope: !2, file: !2, line: 63, type: !5, scopeLine: 63, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !4)
!70 = distinct !DISubprogram(name: "weakKer", scope: !2, file: !2, line: 73, type: !5, scopeLine: 73, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !4)
