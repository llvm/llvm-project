; REQUIRES: x86-registered-target
; RUN: llc -filetype=obj %s -o %t
; RUN: llvm-profgen --binary=%t --perfscript=%s --output=%t1 --show-disassembly-only -x86-asm-syntax=intel --show-source-locations | FileCheck %s --match-full-lines
; RUN: llvm-profgen --binary=%t --perfscript=%s --output=%t2 --show-disassembly-only -x86-asm-syntax=intel --show-source-locations --show-canonical-fname | FileCheck %s --match-full-lines  --check-prefix=CHECK-CANO

; CHECK: Disassembly of section .text [0x0, 0x4a]:
; CHECK: <funcA.llvm.1000>:
; CHECK:        0:	mov	eax, edi                         funcA.llvm.1000:0
; CHECK:        2:	mov	ecx, dword ptr [rip]             funcLeaf:2 @ funcA.llvm.1000:1
; CHECK:        8:	lea	edx, [rcx + 3]                   fib:2 @ funcLeaf:2 @ funcA.llvm.1000:1
; CHECK:        b:	cmp	ecx, 3                           fib:2 @ funcLeaf:2 @ funcA.llvm.1000:1
; CHECK:        e:	cmovl	edx, ecx                       fib:2 @ funcLeaf:2 @ funcA.llvm.1000:1
; CHECK:       11:	sub	eax, edx                         funcLeaf:2 @ funcA.llvm.1000:1
; CHECK:       13:	ret                                  funcA.llvm.1000:2
; CHECK:       14:	nop	word ptr cs:[rax + rax]
; CHECK:       1e:	nop
; CHECK-CANO: <funcA>:
; CHECK-CANO:        0:	mov	eax, edi                         funcA:0
; CHECK-CANO:        2:	mov	ecx, dword ptr [rip]             funcLeaf:2 @ funcA:1
; CHECK-CANO:        8:	lea	edx, [rcx + 3]                   fib:2 @ funcLeaf:2 @ funcA:1
; CHECK-CANO:        b:	cmp	ecx, 3                           fib:2 @ funcLeaf:2 @ funcA:1
; CHECK-CANO:        e:	cmovl	edx, ecx                       fib:2 @ funcLeaf:2 @ funcA:1
; CHECK-CANO:       11:	sub	eax, edx                         funcLeaf:2 @ funcA:1
; CHECK-CANO:       13:	ret                                  funcA:2
; CHECK-CANO:       14:	nop	word ptr cs:[rax + rax]
; CHECK-CANO:       1e:	nop
; CHECK: <funcLeaf>:
; CHECK:      20:	mov	eax, edi                           funcLeaf:1
; CHECK:      22:	mov	ecx, dword ptr [rip]               funcLeaf:2
; CHECK:      28:	lea	edx, [rcx + 3]                     fib:2 @ funcLeaf:2
; CHECK:      2b:	cmp	ecx, 3                             fib:2 @ funcLeaf:2
; CHECK:      2e:	cmovl	edx, ecx                         fib:2 @ funcLeaf:2
; CHECK:      31:	sub	eax, edx                           funcLeaf:2
; CHECK:      33:	ret                                    funcLeaf:3
; CHECK:      34:	nop	word ptr cs:[rax + rax]
; CHECK:      3e:	nop
; CHECK: <fib>:
; CHECK:      40:	lea	eax, [rdi + 3]                     fib:2
; CHECK:      43:	cmp	edi, 3                             fib:2
; CHECK:      46:	cmovl	eax, edi                         fib:2
; CHECK:      49:	ret                                    fib:8

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@factor = dso_local global i32 3

define dso_local i32 @funcA.llvm.1000(i32 %x) !dbg !12 {
entry:
  call void @llvm.dbg.value(metadata i32 %x, metadata !16, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 %x, metadata !19, metadata !DIExpression()), !dbg !22
  %0 = load volatile i32, i32* @factor, align 4, !dbg !24, !tbaa !25
  call void @llvm.dbg.value(metadata i32 %0, metadata !29, metadata !DIExpression()), !dbg !32
  %cmp.i.i = icmp slt i32 %0, 3, !dbg !34
  %add.i.i = add nsw i32 %0, 3, !dbg !36
  %retval.0.i.i = select i1 %cmp.i.i, i32 %0, i32 %add.i.i, !dbg !36
  %sub.i = sub nsw i32 %x, %retval.0.i.i, !dbg !37
  call void @llvm.dbg.value(metadata i32 %sub.i, metadata !19, metadata !DIExpression()), !dbg !22
  call void @llvm.dbg.value(metadata i32 %sub.i, metadata !17, metadata !DIExpression()), !dbg !18
  ret i32 %sub.i, !dbg !38
}

define dso_local i32 @funcLeaf(i32 %x) !dbg !20 {
entry:
  call void @llvm.dbg.value(metadata i32 %x, metadata !19, metadata !DIExpression()), !dbg !39
  %0 = load volatile i32, i32* @factor, align 4, !dbg !40, !tbaa !25
  call void @llvm.dbg.value(metadata i32 %0, metadata !29, metadata !DIExpression()), !dbg !41
  %cmp.i = icmp slt i32 %0, 3, !dbg !43
  %add.i = add nsw i32 %0, 3, !dbg !44
  %retval.0.i = select i1 %cmp.i, i32 %0, i32 %add.i, !dbg !44
  %sub = sub nsw i32 %x, %retval.0.i, !dbg !45
  call void @llvm.dbg.value(metadata i32 %sub, metadata !19, metadata !DIExpression()), !dbg !39
  ret i32 %sub, !dbg !46
}

define dso_local i32 @fib(i32 %x) !dbg !30 {
entry:
  call void @llvm.dbg.value(metadata i32 %x, metadata !29, metadata !DIExpression()), !dbg !47
  %cmp = icmp slt i32 %x, 3, !dbg !48
  %add = add nsw i32 %x, 3, !dbg !49
  %retval.0 = select i1 %cmp, i32 %x, i32 %add, !dbg !49
  ret i32 %retval.0, !dbg !50
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #2

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10}

!1 = distinct !DIGlobalVariable(name: "factor", scope: !2, file: !3, line: 3, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, emissionKind: FullDebug)
!3 = !DIFile(filename: "test.c", directory: "test")
!4 = !{}
!6 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !7)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!12 = distinct !DISubprogram(name: "funcA.llvm.1000", scope: !3, file: !3, line: 6, type: !13, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{!7, !7}
!15 = !{!16, !17}
!16 = !DILocalVariable(name: "x", arg: 1, scope: !12, file: !3, line: 6, type: !7)
!17 = !DILocalVariable(name: "r", scope: !12, file: !3, line: 7, type: !7)
!18 = !DILocation(line: 0, scope: !12)
!19 = !DILocalVariable(name: "x", arg: 1, scope: !20, file: !3, line: 22, type: !7)
!20 = distinct !DISubprogram(name: "funcLeaf", scope: !3, file: !3, line: 22, type: !13, scopeLine: 23, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !21)
!21 = !{!19}
!22 = !DILocation(line: 0, scope: !20, inlinedAt: !23)
!23 = distinct !DILocation(line: 7, column: 11, scope: !12)
!24 = !DILocation(line: 24, column: 12, scope: !20, inlinedAt: !23)
!25 = !{!26, !26, i64 0}
!26 = !{!"int", !27, i64 0}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C/C++ TBAA"}
!29 = !DILocalVariable(name: "x", arg: 1, scope: !30, file: !3, line: 11, type: !7)
!30 = distinct !DISubprogram(name: "fib", scope: !3, file: !3, line: 11, type: !13, scopeLine: 12, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !31)
!31 = !{!29}
!32 = !DILocation(line: 0, scope: !30, inlinedAt: !33)
!33 = distinct !DILocation(line: 24, column: 8, scope: !20, inlinedAt: !23)
!34 = !DILocation(line: 13, column: 9, scope: !35, inlinedAt: !33)
!35 = distinct !DILexicalBlock(scope: !30, file: !3, line: 13, column: 7)
!36 = !DILocation(line: 13, column: 7, scope: !30, inlinedAt: !33)
!37 = !DILocation(line: 24, column: 5, scope: !20, inlinedAt: !23)
!38 = !DILocation(line: 8, column: 3, scope: !12)
!39 = !DILocation(line: 0, scope: !20)
!40 = !DILocation(line: 24, column: 12, scope: !20)
!41 = !DILocation(line: 0, scope: !30, inlinedAt: !42)
!42 = distinct !DILocation(line: 24, column: 8, scope: !20)
!43 = !DILocation(line: 13, column: 9, scope: !35, inlinedAt: !42)
!44 = !DILocation(line: 13, column: 7, scope: !30, inlinedAt: !42)
!45 = !DILocation(line: 24, column: 5, scope: !20)
!46 = !DILocation(line: 25, column: 3, scope: !20)
!47 = !DILocation(line: 0, scope: !30)
!48 = !DILocation(line: 13, column: 9, scope: !35)
!49 = !DILocation(line: 13, column: 7, scope: !30)
!50 = !DILocation(line: 19, column: 1, scope: !30)
