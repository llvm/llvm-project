; RUN: llc -O3 -stop-after virtregrewriter %s -o - | FileCheck %s

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "ppc64"

; Verify that live-debug-variables includes the stack slot offset
; for the sub-register in the debug expression for `spilled'.

; This reproducer was generated from the following C file:
;
;     extern int foo(void);
;     extern void bar(int);
;
;     int main() {
;       int spilled = foo();
;
;       // Clobber all GPRs.
;       __asm volatile ("" : : : "0", "1", "2", "3", "4", "5", "6", "7",
;                                "8", "9", "10", "11", "12", "13", "14", "15",
;                                "16", "17", "18", "19", "20", "21", "22", "23",
;                                "24", "25", "26", "27", "28", "29", "30", "31");
;
;       for(;;)
;         bar(spilled);
;
;       return 0;
;     }
;
; compiled using:
;
;     clang --target=ppc64 -O3 -g -S -emit-llvm

; CHECK: ![[VAR:.*]] = !DILocalVariable(name: "spilled"

; CHECK: STD $x3, 0, %stack.0
; CHECK-NEXT: DBG_VALUE %stack.0, 0, ![[VAR]], !DIExpression(DW_OP_plus_uconst, 4)

; Function Attrs: noreturn nounwind
define signext i32 @main() local_unnamed_addr #0 !dbg !8 {
entry:
  %call = tail call signext i32 @foo() #2, !dbg !14
  call void @llvm.dbg.value(metadata i32 %call, metadata !13, metadata !DIExpression()), !dbg !14
  tail call void asm sideeffect "", "~{r0},~{r1},~{r2},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31}"() #2, !dbg !14, !srcloc !15
  br label %for.cond, !dbg !14

for.cond:                                         ; preds = %for.cond, %entry
  tail call void @bar(i32 signext %call) #2, !dbg !14
  br label %for.cond, !dbg !14
}

declare signext i32 @foo() local_unnamed_addr

declare void @bar(i32 signext) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { noreturn nounwind }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "live-debug-vars-subreg-offset.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 8.0.0"}
!8 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 4, type: !9, isLocal: false, isDefinition: true, scopeLine: 4, isOptimized: true, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DILocalVariable(name: "spilled", scope: !8, file: !1, line: 5, type: !11)
!14 = !DILocation(line: 5, column: 17, scope: !8)
!15 = !{i32 125}
