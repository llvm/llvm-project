; Check info on linkage.

; RUN: opt -pass-remarks=kernel-info -passes=kernel-info \
; RUN:     -disable-output %s 2>&1 | \
; RUN:   FileCheck -match-full-lines %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; CHECK: remark: test.c:3:0: in function 'f', ExternalNotKernel = 1
define external void @f() !dbg !10 {
entry:
  ret void
}

; CHECK: remark: test.c:13:0: in artificial function 'g', ExternalNotKernel = 1
define void @g() !dbg !20 {
entry:
  ret void
}

; CHECK: remark: test.c:23:0: in function 'h', ExternalNotKernel = 0
define external void @h() #0 !dbg !30 {
entry:
  ret void
}

; CHECK: remark: test.c:33:0: in artificial function 'i', ExternalNotKernel = 0
define weak void @i() !dbg !40 {
entry:
  ret void
}

attributes #0 = { "kernel" }

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!2 = !DIFile(filename: "test.c", directory: "/tmp")
!3 = !{null}
!4 = !{}
!10 = distinct !DISubprogram(name: "f", scope: !2, file: !2, line: 3, type: !11, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !4)
!11 = !DISubroutineType(types: !3)
!20 = distinct !DISubprogram(name: "g", scope: !2, file: !2, line: 13, type: !21, scopeLine: 13, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1, retainedNodes: !4)
!21 = distinct !DISubroutineType(types: !3)
!30 = distinct !DISubprogram(name: "h", scope: !2, file: !2, line: 23, type: !31, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !4)
!31 = distinct !DISubroutineType(types: !3)
!40 = distinct !DISubprogram(name: "i", scope: !2, file: !2, line: 33, type: !41, scopeLine: 33, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1, retainedNodes: !4)
!41 = distinct !DISubroutineType(types: !3)
