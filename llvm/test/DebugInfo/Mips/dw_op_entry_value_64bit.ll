;; Test mips64:
; RUN: llc -emit-call-site-info -stop-after=livedebugvalues -mtriple=mips64-linux-gnu -o - %s | FileCheck %s --check-prefix=CHECK64
; RUN: llc -force-instr-ref-livedebugvalues=1 -emit-call-site-info -stop-after=livedebugvalues -mtriple=mips64-linux-gnu -o - %s | FileCheck %s --check-prefix=CHECK64
;; Test mips64el:
; RUN: llc -emit-call-site-info -stop-after=livedebugvalues -mtriple=mips64el-linux-gnu -o - %s | FileCheck %s --check-prefix=CHECK64el
; RUN: llc -force-instr-ref-livedebugvalues=1 -emit-call-site-info -stop-after=livedebugvalues -mtriple=mips64el-linux-gnu -o - %s | FileCheck %s --check-prefix=CHECK64el

;; Built from source:
;; extern long fn1(long,long,long);
;; long fn2(long a, long b, long c) {
;;   long local = fn1(a+b, c, b+10);
;;   if (local > 10)
;;     return local + 10;
;;   return b;
;; }
;; Using command:
;; clang -g -O2 -target mips64-linux-gnu m.c -c -S -emit-llvm
;; Confirm that info from callSites attribute is used as entry_value in DIExpression.

;; Test mips64:
; CHECK64: $a0_64 = nsw DADDu $a1_64, killed renamable $a0_64,
; CHECK64-NEXT: DBG_VALUE $a0_64, $noreg, !15, !DIExpression(DW_OP_LLVM_entry_value, 1)

;; Test mips64el:
; CHECK64el: $a0_64 = nsw DADDu $a1_64, killed renamable $a0_64,
; CHECK64el-NEXT: DBG_VALUE $a0_64, $noreg, !15, !DIExpression(DW_OP_LLVM_entry_value, 1)

; ModuleID = 'm.c'
source_filename = "m.c"
target datalayout = "E-m:e-i8:8:32-i16:16:32-i64:64-n32:64-S128"
target triple = "mips64-unknown-linux-gnu"

; Function Attrs: nounwind
define i64 @fn2(i64 signext %a, i64 signext %b, i64 signext %c) local_unnamed_addr !dbg !13 {
entry:
  call void @llvm.dbg.value(metadata i64 %a, metadata !15, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata i64 %b, metadata !16, metadata !DIExpression()), !dbg !19
  call void @llvm.dbg.value(metadata i64 %c, metadata !17, metadata !DIExpression()), !dbg !19
  %add = add nsw i64 %b, %a, !dbg !19
  %add1 = add nsw i64 %b, 10, !dbg !19
  %call = tail call i64 @fn1(i64 signext %add, i64 signext %c, i64 signext %add1), !dbg !19
  call void @llvm.dbg.value(metadata i64 %call, metadata !18, metadata !DIExpression()), !dbg !19
  %cmp = icmp sgt i64 %call, 10, !dbg !23
  %add2 = add nsw i64 %call, 10, !dbg !19
  %retval.0 = select i1 %cmp, i64 %add2, i64 %b, !dbg !19
  ret i64 %retval.0, !dbg !19
}

declare !dbg !4 i64 @fn1(i64 signext, i64 signext, i64 signext) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "m.c", directory: "/dir")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "fn1", scope: !1, file: !1, line: 1, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7, !7, !7}
!7 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 7, !"PIC Level", i32 1}
!12 = !{!"clang version 11.0.0"}
!13 = distinct !DISubprogram(name: "fn2", scope: !1, file: !1, line: 2, type: !5, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!14 = !{!15, !16, !17, !18}
!15 = !DILocalVariable(name: "a", arg: 1, scope: !13, file: !1, line: 2, type: !7)
!16 = !DILocalVariable(name: "b", arg: 2, scope: !13, file: !1, line: 2, type: !7)
!17 = !DILocalVariable(name: "c", arg: 3, scope: !13, file: !1, line: 2, type: !7)
!18 = !DILocalVariable(name: "local", scope: !13, file: !1, line: 3, type: !7)
!19 = !DILocation(line: 0, scope: !13)
!23 = !DILocation(line: 4, column: 13, scope: !24)
!24 = distinct !DILexicalBlock(scope: !13, file: !1, line: 4, column: 7)
