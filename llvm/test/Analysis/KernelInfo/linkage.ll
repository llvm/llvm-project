; Check info on linkage.

; RUN: opt -pass-remarks=kernel-info -passes=kernel-info \
; RUN:     -disable-output %s 2>&1 | \
; RUN:   FileCheck -match-full-lines %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; CHECK: remark: test.c:3:0: in function 'extNotKer', ExternalNotKernel = 1
define external void @extNotKer() !dbg !10 {
entry:
  ret void
}

; CHECK: remark: test.c:13:0: in artificial function 'impNotKer', ExternalNotKernel = 1
define void @impNotKer() !dbg !20 {
entry:
  ret void
}

; CHECK: remark: test.c:23:0: in artificial function 'weakNotKer', ExternalNotKernel = 0
define weak void @weakNotKer() !dbg !30 {
entry:
  ret void
}

; CHECK: remark: test.c:33:0: in function 'extKerAttr', ExternalNotKernel = 0
define external void @extKerAttr() #0 !dbg !40 {
entry:
  ret void
}

; CHECK: remark: test.c:43:0: in function 'extKer', ExternalNotKernel = 0
define external void @extKer() !dbg !50 {
entry:
  ret void
}

; CHECK: remark: test.c:53:0: in artificial function 'weakKer', ExternalNotKernel = 0
define weak void @weakKer() !dbg !60 {
entry:
  ret void
}

attributes #0 = { "kernel" }

!nvvm.annotations = !{!42, !52, !62}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!2 = !DIFile(filename: "test.c", directory: "/tmp")
!3 = !{null}
!4 = !{}

!10 = distinct !DISubprogram(name: "extNotKer", scope: !2, file: !2, line: 3, type: !11, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !4)
!11 = !DISubroutineType(types: !3)

!20 = distinct !DISubprogram(name: "impNotKer", scope: !2, file: !2, line: 13, type: !21, scopeLine: 13, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1, retainedNodes: !4)
!21 = distinct !DISubroutineType(types: !3)

!30 = distinct !DISubprogram(name: "weakNotKer", scope: !2, file: !2, line: 23, type: !31, scopeLine: 23, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1, retainedNodes: !4)
!31 = distinct !DISubroutineType(types: !3)

!40 = distinct !DISubprogram(name: "extKerAttr", scope: !2, file: !2, line: 33, type: !41, scopeLine: 33, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !4)
!41 = distinct !DISubroutineType(types: !3)
!42 = !{ptr @extKerAttr, !"kernel", i32 1}

!50 = distinct !DISubprogram(name: "extKer", scope: !2, file: !2, line: 43, type: !51, scopeLine: 43, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !4)
!51 = distinct !DISubroutineType(types: !3)
!52 = !{ptr @extKer, !"kernel", i32 1}

!60 = distinct !DISubprogram(name: "weakKer", scope: !2, file: !2, line: 53, type: !61, scopeLine: 53, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1, retainedNodes: !4)
!61 = distinct !DISubroutineType(types: !3)
!62 = !{ptr @weakKer, !"kernel", i32 1}
