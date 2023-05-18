;; Test mips:
; RUN: llc -mtriple=mips-linux-gnu -emit-call-site-info %s -stop-after=finalize-isel -o -| FileCheck %s
;; Test mipsel:
; RUN: llc -mtriple=mipsel-linux-gnu -emit-call-site-info %s -stop-after=finalize-isel -o -| FileCheck %s

;; Verify that call site info is not emitted for parameter passed through 64-bit register $d
;; which splits into two 32-bit physical regs.
;; Source:
;; extern double bar(double,int);
;; double foo(double self){
;;   int b = 1;
;;   double a = bar(self,b);
;;   return a;
;; }

;; Test mips and mipsel:
; CHECK: name:            foo
; CHECK: callSites:
; CHECK-NEXT: bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; CHECK-NOT:   arg: 0, reg: '$a0'
; CHECK-NOT:   arg: 0, reg: '$d6'
; CHECK-NEXT:   arg: 1, reg: '$a2'

; ModuleID = 'm.c'
source_filename = "m.c"
target datalayout = "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"
target triple = "mips-unknown-linux-gnu"

; Function Attrs: nounwind
define dso_local double @foo(double %self) local_unnamed_addr !dbg !13 {
entry:
  call void @llvm.dbg.value(metadata double %self, metadata !17, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 1, metadata !18, metadata !DIExpression()), !dbg !20
  %call = tail call double @bar(double %self, i32 signext 1), !dbg !20
  call void @llvm.dbg.value(metadata double %call, metadata !19, metadata !DIExpression()), !dbg !20
  ret double %call, !dbg !20
}

declare !dbg !4 dso_local double @bar(double, i32 signext) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "m.c", directory: "/dir")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "bar", scope: !1, file: !1, line: 1, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7, !8}
!7 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"clang version 11.0.0"}
!13 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !14, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!14 = !DISubroutineType(types: !15)
!15 = !{!7, !7}
!16 = !{!17, !18, !19}
!17 = !DILocalVariable(name: "self", arg: 1, scope: !13, file: !1, line: 3, type: !7)
!18 = !DILocalVariable(name: "b", scope: !13, file: !1, line: 4, type: !8)
!19 = !DILocalVariable(name: "a", scope: !13, file: !1, line: 5, type: !7)
!20 = !DILocation(line: 0, scope: !13)
