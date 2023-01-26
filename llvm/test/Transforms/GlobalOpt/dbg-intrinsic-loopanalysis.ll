; RUN: opt -passes="globalopt" < %s -o /dev/null -debug-pass-manager 2>&1 | FileCheck %s
; RUN: opt -strip-debug -S < %s | opt -passes="globalopt" -o /dev/null -debug-pass-manager 2>&1 | FileCheck %s

; Make sure that the call to dbg.declare does not prevent running BlockFrequency
; and (especially) Loop Analysis.
; Later passes (e.g. instcombine) may behave in different ways depending on if
; LoopInfo is available or not. Therefore, letting GlobalOpt run or not run
; LoopAnalysis depending on the presence of a dbg.declare may make the compiler
; generate different code with and without debug info.

; CHECK: Running pass: GlobalOptPass on [module]
; CHECK: Running analysis: BlockFrequencyAnalysis on h
; CHECK: Running analysis: LoopAnalysis on h

define i16 @h(ptr %k) {
entry:
  call void @llvm.dbg.declare(metadata ptr %k, metadata !1, metadata !DIExpression()), !dbg !22
  %call = call i16 @gaz()
  ret i16 %call
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)  #0

define internal i16 @gaz() {
entry:
  ret i16 0
}

attributes #0 = { nocallback nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!17}
!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DILocalVariable(name: "k", arg: 1, scope: !2, file: !3, line: 13, type: !19)
!2 = distinct !DISubprogram(name: "h", scope: !3, file: !3, line: 13, type: !4, scopeLine: 13, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !17, retainedNodes: !21)
!3 = !DIFile(filename: "foo2.c", directory: "/bar")
!4 = !DISubroutineType(types: !5)
!5 = !{!19}
!17 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 16", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !18, globals: !20, splitDebugInlining: false, nameTableKind: None)
!18 = !{!19}
!19 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!20 = !{}
!21 = !{!1}
!22 = !DILocation(line: 13, column: 27, scope: !2)
