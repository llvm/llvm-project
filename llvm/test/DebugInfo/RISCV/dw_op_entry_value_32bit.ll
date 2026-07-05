;; Test RISC-V 32 bit:
; RUN: llc -emit-call-site-info -stop-after=livedebugvalues -mtriple=riscv32-linux-gnu -o - %s | FileCheck %s --check-prefix=CHECK32

;; Built from source:
;; extern long fn1(long,long,long);
;; long fn2(long a, long b, long c) {
;;   long local = fn1(a+b, c, b+10);
;;   if (local > 10)
;;     return local + 10;
;;   return b;
;; }
;; Using command:
;; clang -g -O2 -target riscv32-linux-gnu m.c -c -S -emit-llvm
;; Confirm that info from callSites attribute is used as entry_value in DIExpression.

;; Test riscv32:
; CHECK32: $x10 = nsw ADD $x11, killed renamable $x10
; CHECK32-NEXT: DBG_VALUE $x10, $noreg, !{{.*}}, !DIExpression(DW_OP_LLVM_entry_value, 1)

; ModuleID = 'm.c'
source_filename = "m.c"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @fn2(i32 noundef %a, i32 noundef %b, i32 noundef %c) !dbg !14 {
entry:
    #dbg_value(i32 %a, !20, !DIExpression(), !23)
    #dbg_value(i32 %b, !21, !DIExpression(), !23)
    #dbg_value(i32 %c, !22, !DIExpression(), !23)
  %add = add nsw i32 %b, %a
  %add1 = add nsw i32 %b, 10
  %call = tail call i32 @fn1(i32 noundef %add, i32 noundef %c, i32 noundef %add1)
    #dbg_value(i32 %call, !22, !DIExpression(), !23)
  %cmp = icmp sgt i32 %call, 10
  %add2 = add nuw nsw i32 %call, 10
  %retval.0 = select i1 %cmp, i32 %add2, i32 %b
  ret i32 %retval.0, !dbg !29
}

declare !dbg !30 i32 @fn1(i32 noundef, i32 noundef, i32 noundef)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "m.c", directory: ".")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang"}
!14 = distinct !DISubprogram(name: "fn2", scope: !1, file: !1, line: 2, type: !15, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !18)
!15 = !DISubroutineType(types: !16)
!16 = !{!17, !17, !17, !17}
!17 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!18 = !{!19, !20, !21, !22}
!19 = !DILocalVariable(name: "a", arg: 1, scope: !14, file: !1, line: 2, type: !17)
!20 = !DILocalVariable(name: "b", arg: 2, scope: !14, file: !1, line: 2, type: !17)
!21 = !DILocalVariable(name: "c", arg: 3, scope: !14, file: !1, line: 2, type: !17)
!22 = !DILocalVariable(name: "local", scope: !14, file: !1, line: 3, type: !17)
!23 = !DILocation(line: 0, scope: !14)
!29 = !DILocation(line: 7, column: 1, scope: !14)
!30 = !DISubprogram(name: "fn1", scope: !1, file: !1, line: 1, type: !15, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)

