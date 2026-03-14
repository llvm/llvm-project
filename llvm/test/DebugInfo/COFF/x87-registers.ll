; RUN: llc -experimental-debug-variable-locations=true < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s

; CHECK:      DefRangeRegisterSym {
; CHECK-NEXT:   Kind: S_DEFRANGE_REGISTER
; CHECK-NEXT:   Register: ST0

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

define i32 @a() !dbg !8 {
entry:
  call void @llvm.dbg.declare(metadata ptr undef, metadata !13, metadata !DIExpression(DW_OP_LLVM_fragment, 80, 48)), !dbg !15
  %0 = tail call x86_fp80 asm sideeffect "", "={st},~{dirflag},~{fpsr},~{flags}"(), !dbg !16, !srcloc !17
  call void @llvm.dbg.value(metadata x86_fp80 %0, metadata !13, metadata !DIExpression()), !dbg !18
  ret i32 undef, !dbg !19
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "codeview-fp0.c", directory: "llvm/build")
!2 = !{i32 2, !"CodeView", i32 1}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 2}
!5 = !{i32 7, !"PIC Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 1}
!7 = !{!"clang version 14.0.0"}
!8 = distinct !DISubprogram(name: "a", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DILocalVariable(name: "b", scope: !8, file: !1, line: 2, type: !14)
!14 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!15 = !DILocation(line: 2, scope: !8)
!16 = !DILocation(line: 3, scope: !8)
!17 = !{i64 40}
!18 = !DILocation(line: 0, scope: !8)
!19 = !DILocation(line: 4, scope: !8)
