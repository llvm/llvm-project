; RUN: opt < %s -scalarrepl -S | FileCheck %s
; RUN: opt < %s -scalarrepl-ssa -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.0"

; CHECK: f
; CHECK-NOT: llvm.dbg.declare
; CHECK: llvm.dbg.value
; CHECK: llvm.dbg.value
; CHECK: llvm.dbg.value
; CHECK: llvm.dbg.value
; CHECK: llvm.dbg.value

define i32 @f(i32 %a, i32 %b) nounwind ssp !dbg !1 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !6, metadata !DIExpression()), !dbg !7
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !8, metadata !DIExpression()), !dbg !9
  call void @llvm.dbg.declare(metadata i32* %c, metadata !10, metadata !DIExpression()), !dbg !12
  %tmp = load i32, i32* %a.addr, align 4, !dbg !13
  store i32 %tmp, i32* %c, align 4, !dbg !13
  %tmp1 = load i32, i32* %a.addr, align 4, !dbg !14
  %tmp2 = load i32, i32* %b.addr, align 4, !dbg !14
  %add = add nsw i32 %tmp1, %tmp2, !dbg !14
  store i32 %add, i32* %a.addr, align 4, !dbg !14
  %tmp3 = load i32, i32* %c, align 4, !dbg !15
  %tmp4 = load i32, i32* %b.addr, align 4, !dbg !15
  %sub = sub nsw i32 %tmp3, %tmp4, !dbg !15
  store i32 %sub, i32* %b.addr, align 4, !dbg !15
  %tmp5 = load i32, i32* %a.addr, align 4, !dbg !16
  %tmp6 = load i32, i32* %b.addr, align 4, !dbg !16
  %add7 = add nsw i32 %tmp5, %tmp6, !dbg !16
  ret i32 %add7, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk 131941)", isOptimized: false, emissionKind: 0, file: !18, enums: !19, retainedTypes: !19, subprograms: !17)
!1 = distinct !DISubprogram(name: "f", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 1, file: !18, scope: !2, type: !3)
!2 = !DIFile(filename: "/d/j/debug-test.c", directory: "/Volumes/Data/b")
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !DILocalVariable(name: "a", line: 1, arg: 1, scope: !1, file: !2, type: !5)
!7 = !DILocation(line: 1, column: 11, scope: !1)
!8 = !DILocalVariable(name: "b", line: 1, arg: 2, scope: !1, file: !2, type: !5)
!9 = !DILocation(line: 1, column: 18, scope: !1)
!10 = !DILocalVariable(name: "c", line: 2, scope: !11, file: !2, type: !5)
!11 = distinct !DILexicalBlock(line: 1, column: 21, file: !18, scope: !1)
!12 = !DILocation(line: 2, column: 9, scope: !11)
!13 = !DILocation(line: 2, column: 14, scope: !11)
!14 = !DILocation(line: 3, column: 5, scope: !11)
!15 = !DILocation(line: 4, column: 5, scope: !11)
!16 = !DILocation(line: 5, column: 5, scope: !11)
!17 = !{!1}
!18 = !DIFile(filename: "/d/j/debug-test.c", directory: "/Volumes/Data/b")
!19 = !{}
!20 = !{i32 1, !"Debug Info Version", i32 3}
