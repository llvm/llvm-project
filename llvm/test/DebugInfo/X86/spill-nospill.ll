; RUN: llc < %s | FileCheck %s
; RUN: llc < %s -filetype=obj | llvm-dwarfdump - | FileCheck %s --check-prefix=DWARF

; This test creates two UserValues in LiveDebugVariables with one location
; each. x must be spilled, but y will be allocated to a CSR. x's location
; should be indirect, but y's should be direct.

; C source:
; #define FORCE_SPILL() \
;   __asm volatile("" : : : \
;                    "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "r8", \
;                    "r9", "r10", "r11", "r12", "r13", "r14", "r15")
; int g(int);
; int f() {
;   int x = g(0);
;   FORCE_SPILL();
;   int y = g(0);
;   g(y);
;   g(y);
;   g(y);
;   return x + y;
; }

; CHECK-LABEL: f: # @f
; CHECK: callq   g
; CHECK: movl    %eax, [[X_OFFS:[0-9]+]](%rsp)          # 4-byte Spill
; CHECK: #DEBUG_VALUE: f:x <- [DW_OP_plus_uconst [[X_OFFS]]] [$rsp+0]
; CHECK: #APP
; CHECK: #NO_APP
; CHECK: callq   g
; CHECK: movl    %eax, %[[CSR:[^ ]*]]
; CHECK: #DEBUG_VALUE: f:y <- $esi
; CHECK: movl    %eax, %ecx
; CHECK: callq   g
; CHECK: movl    %[[CSR]], %ecx
; CHECK: callq   g
; CHECK: movl    %[[CSR]], %ecx
; CHECK: callq   g
; CHECK: movl    [[X_OFFS]](%rsp), %eax          # 4-byte Reload
; CHECK: #DEBUG_VALUE: f:x <- $eax
; CHECK: addl    %[[CSR]], %eax

; DWARF:      DW_TAG_variable
; DWARF-NEXT:   DW_AT_location        (
; DWARF-NEXT:      [{{.*}}, {{.*}}): DW_OP_breg7 RSP+36
; DWARF-NEXT:      [{{.*}}, {{.*}}): DW_OP_reg0 RAX)
; DWARF-NEXT:   DW_AT_name    ("x")

; DWARF:      DW_TAG_variable
; DWARF-NEXT:   DW_AT_location        (
; DWARF-NEXT:      [{{.*}},  {{.*}}): DW_OP_reg4 RSI)
; DWARF-NEXT:   DW_AT_name    ("y")

; ModuleID = 'spill-nospill.c'
source_filename = "spill-nospill.c"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

; Function Attrs: nounwind uwtable
define i32 @f() local_unnamed_addr #0 !dbg !8 {
entry:
  %call = tail call i32 @g(i32 0) #3, !dbg !15
  tail call void @llvm.dbg.value(metadata i32 %call, metadata !13, metadata !DIExpression()), !dbg !16
  tail call void asm sideeffect "", "~{rax},~{rbx},~{rcx},~{rdx},~{rsi},~{rdi},~{rbp},~{r8},~{r9},~{r10},~{r11},~{r12},~{r13},~{r14},~{r15},~{dirflag},~{fpsr},~{flags}"() #3, !dbg !17, !srcloc !18
  %call1 = tail call i32 @g(i32 0) #3, !dbg !19
  tail call void @llvm.dbg.value(metadata i32 %call1, metadata !14, metadata !DIExpression()), !dbg !20
  %call2 = tail call i32 @g(i32 %call1) #3, !dbg !21
  %call3 = tail call i32 @g(i32 %call1) #3, !dbg !22
  %call4 = tail call i32 @g(i32 %call1) #3, !dbg !23
  %add = add nsw i32 %call1, %call, !dbg !24
  ret i32 %add, !dbg !25
}

declare i32 @g(i32) local_unnamed_addr #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "spill-nospill.c", directory: "C:\5Csrc\5Cllvm-project\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 6.0.0 "}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 8, type: !9, isLocal: false, isDefinition: true, scopeLine: 8, isOptimized: true, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13, !14}
!13 = !DILocalVariable(name: "x", scope: !8, file: !1, line: 9, type: !11)
!14 = !DILocalVariable(name: "y", scope: !8, file: !1, line: 11, type: !11)
!15 = !DILocation(line: 9, column: 11, scope: !8)
!16 = !DILocation(line: 9, column: 7, scope: !8)
!17 = !DILocation(line: 10, column: 3, scope: !8)
!18 = !{i32 -2147472112}
!19 = !DILocation(line: 11, column: 11, scope: !8)
!20 = !DILocation(line: 11, column: 7, scope: !8)
!21 = !DILocation(line: 12, column: 3, scope: !8)
!22 = !DILocation(line: 13, column: 3, scope: !8)
!23 = !DILocation(line: 14, column: 3, scope: !8)
!24 = !DILocation(line: 15, column: 12, scope: !8)
!25 = !DILocation(line: 15, column: 3, scope: !8)
