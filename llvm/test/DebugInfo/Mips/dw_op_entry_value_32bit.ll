;; Test mips32:
; RUN: llc -emit-call-site-info -stop-after=livedebugvalues -mtriple=mips-linux-gnu -o - %s | FileCheck %s
;; Test mipsel:
; RUN: llc -emit-call-site-info -stop-after=livedebugvalues -mtriple=mipsel-linux-gnu -o - %s | FileCheck %s --check-prefix=CHECKel

;; Built from source:
;; extern long fn1(long,long,long);
;; long fn2(long a, long b, long c) {
;;   long local = fn1(a+b, c, b+10);
;;   if (local > 10)
;;     return local + 10;
;;   return b;
;; }
;; Using command:
;; clang -g -O2 -target mips-linux-gnu  m.c -c -S -emit-llvm
;; Confirm that info from callSites attribute is used as entry_value in DIExpression.

;; Test mips32:
; CHECK: $a0 = nsw ADDu $a1, killed renamable $a0,
; CHECK-NEXT: DBG_VALUE $a0, $noreg, !14, !DIExpression(DW_OP_LLVM_entry_value, 1)

;; Test mipsel:
; CHECKel: $a0 = nsw ADDu $a1, killed renamable $a0,
; CHECKel-NEXT: DBG_VALUE $a0, $noreg, !14, !DIExpression(DW_OP_LLVM_entry_value, 1)

; ModuleID = 'm.c'
source_filename = "m.c"
target datalayout = "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"
target triple = "mips-unknown-linux-gnu"

; Function Attrs: nounwind
define dso_local i32 @fn2(i32 signext %a, i32 signext %b, i32 signext %c) local_unnamed_addr !dbg !12 {
entry:
  call void @llvm.dbg.value(metadata i32 %a, metadata !14, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 %b, metadata !15, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.value(metadata i32 %c, metadata !16, metadata !DIExpression()), !dbg !18
  %add = add nsw i32 %b, %a, !dbg !18
  %add1 = add nsw i32 %b, 10, !dbg !18
  %call = tail call i32 @fn1(i32 signext %add, i32 signext %c, i32 signext %add1), !dbg !18
  call void @llvm.dbg.value(metadata i32 %call, metadata !17, metadata !DIExpression()), !dbg !18
  %cmp = icmp sgt i32 %call, 10, !dbg !22
  %add2 = add nsw i32 %call, 10, !dbg !18
  %retval.0 = select i1 %cmp, i32 %add2, i32 %b, !dbg !18
  ret i32 %retval.0, !dbg !18
}

declare !dbg !4 dso_local i32 @fn1(i32 signext, i32 signext, i32 signext) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "m.c", directory: "/dir")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "fn1", scope: !1, file: !1, line: 1, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7, !7, !7}
!7 = !DIBasicType(name: "long int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{!"clang version 11.0.0"}
!12 = distinct !DISubprogram(name: "fn2", scope: !1, file: !1, line: 2, type: !5, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!13 = !{!14, !15, !16, !17}
!14 = !DILocalVariable(name: "a", arg: 1, scope: !12, file: !1, line: 2, type: !7)
!15 = !DILocalVariable(name: "b", arg: 2, scope: !12, file: !1, line: 2, type: !7)
!16 = !DILocalVariable(name: "c", arg: 3, scope: !12, file: !1, line: 2, type: !7)
!17 = !DILocalVariable(name: "local", scope: !12, file: !1, line: 3, type: !7)
!18 = !DILocation(line: 0, scope: !12)
!22 = !DILocation(line: 4, column: 13, scope: !23)
!23 = distinct !DILexicalBlock(scope: !12, file: !1, line: 4, column: 7)
