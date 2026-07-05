; RUN: llvm-as -disable-output <%s 2>&1| FileCheck %s
define i32 @_Z3foov() local_unnamed_addr !dbg !9 {
  ret i32 5
}
!llvm.module.flags = !{!2}
!llvm.dbg.cu = !{!5}
!llvm.linker.options = !{}

!2 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !6, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!6 = !DIFile(filename: "t.cpp", directory: "/")
!7 = !{}
; CHECK: function definition may only have a distinct !dbg attachment
; CHECK: warning: ignoring invalid debug info
!9 = !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !6, file: !6, line: 2, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !7)
!11 = !DISubroutineType(types: !7)
