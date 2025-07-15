; RUN: opt -dropped-variable-stats %s -passes='verify' -S | FileCheck %s --check-prefix=NOT-DROPPED
; NOT-DROPPED: Pass Level, Pass Name, Num of Dropped Variables, Func or Module Name
; NOT-DROPPED-NOT: Function, ADCEPass, 1, _Z3bari

; ModuleID = '/tmp/dropped.cpp'
define noundef range(i32 -2147483646, -2147483648) i32 @_Z3bari(i32 noundef %y) local_unnamed_addr #1 !dbg !19 {
    #dbg_value(i32 %y, !15, !DIExpression(), !23)
  %add = add nsw i32 %y, 2,!dbg !25
  ret i32 %add,!dbg !26
}
!llvm.module.flags = !{ !3, !7}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 20.0.0git (git@github.com:llvm/llvm-project.git 7fc8398aaad65c4c29f1511c374d07308e667af5)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!1 = !DIFile(filename: "/tmp/dropped.cpp", directory: "/Users/shubham/Development/llvm-project")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 7, !"frame-pointer", i32 1}
!9 = distinct !DISubprogram( unit: !0, retainedNodes: !14)
!13 = !DIBasicType()
!14 = !{}
!15 = !DILocalVariable( scope: !9, type: !13)
!19 = distinct !DISubprogram( unit: !0, retainedNodes: !20)
!20 = !{}
!23 = !DILocation( scope: !9, inlinedAt: !24)
!24 = distinct !DILocation( scope: !19)
!25 = !DILocation( scope: !19)
!26 = !DILocation( scope: !19)