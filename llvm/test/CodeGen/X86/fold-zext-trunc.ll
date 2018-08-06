; RUN: llc < %s | FileCheck %s -check-prefix=ASM
; RUN: llc < %s -O0 -mtriple=x86_64-unknown-unknown -mcpu=x86-64 -stop-after livedebugvalues -o - | FileCheck %s -check-prefix=MIR
; PR9055
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i686-pc-linux-gnu"

%struct.S0 = type { i32, [2 x i8], [2 x i8], [4 x i8] }

@g_98 = common global %struct.S0 zeroinitializer, align 4

define void @foo() nounwind !dbg !5 {
; ASM: movzbl
; ASM-NOT: movzbl
; ASM: calll
entry:
  %tmp17 = load i8, i8* getelementptr inbounds (%struct.S0, %struct.S0* @g_98, i32 0, i32 1, i32 0), align 4, !dbg !14
  %tmp54 = zext i8 %tmp17 to i32, !dbg !15
  %foo = load i32, i32* bitcast (i8* getelementptr inbounds (%struct.S0, %struct.S0* @g_98, i32 0, i32 1, i32 0) to i32*), align 4, !dbg !16
; MIR: renamable $edi = MOVZX32rr8 renamable $al, debug-location !16
  %conv.i = trunc i32 %foo to i8, !dbg !17

  tail call void @func_12(i32 %tmp54, i8 zeroext %conv.i) #0, !dbg !18
  call void @llvm.dbg.value(metadata i8 %tmp17, metadata !8, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.value(metadata i32 %tmp54, metadata !10, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata i32 %foo, metadata !12, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.value(metadata i8 %conv.i, metadata !13, metadata !DIExpression()), !dbg !17
  ret void, !dbg !19
}

declare void @func_12(i32, i8 zeroext)

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/Users/vsk/src/llvm.org-master/llvm/test/CodeGen/X86/fold-zext-trunc.ll", directory: "/")
!2 = !{}
!3 = !{i32 6}
!4 = !{i32 4}
!5 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !7)
!6 = !DISubroutineType(types: !2)
!7 = !{!8, !10, !12, !13}
!8 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 1, type: !9)
!9 = !DIBasicType(name: "ty8", size: 8, encoding: DW_ATE_unsigned)
!10 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 2, type: !11)
!11 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!12 = !DILocalVariable(name: "3", scope: !5, file: !1, line: 3, type: !11)
!13 = !DILocalVariable(name: "4", scope: !5, file: !1, line: 4, type: !9)
!14 = !DILocation(line: 1, column: 1, scope: !5)
!15 = !DILocation(line: 2, column: 1, scope: !5)
!16 = !DILocation(line: 3, column: 1, scope: !5)
!17 = !DILocation(line: 4, column: 1, scope: !5)
!18 = !DILocation(line: 5, column: 1, scope: !5)
!19 = !DILocation(line: 6, column: 1, scope: !5)
!20 = !{i32 2, !"Debug Info Version", i32 3}
!llvm.module.flags = !{!20}
