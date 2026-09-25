; Test that if a lexical scope has an abstract lexical block DIE, a conrete lexical
; block DIE for it is emitted even when scope's concrete entities are not emitted
; (for example, when a local variable inside the scope is optimized away).

; This is to prevent debugger from showing wrong values of variables with the same
; name declared in nested scopes.

; Consider shadow.cpp from the commentary below: variable 'x' inside a compound
; statement is fully optimized away, therefore, the lexical block where it is
; declared has no local entities. If that block is not emitted, debugger prints
; the value of shadowed live variable 'x' when `print x` is requested at line with
; `keep(a);`.
; With conrete lexical block for the compund statement emitted, debugger can
; follow block's abstract origin, and discover abstract definition of 'x' that
; shadows definition of outer 'x' at `keep(a)`. Thus, it can faithfully report
; that value of 'x' at `keep(a);` is optimized away.

; IR generated and reduced from:
; $ cat shadow.cpp
; #include <stdio.h>
; #include <math.h>
;
; __attribute__((noinline))
; extern void keep(int) {
;   printf("hello\n");
; }
;
; __attribute__((always_inline))
; int f(int a) {
;   // Live x.
;   int x = a;
;   {
;     // Optimized away x.
;     int x = -abs(a);
;     if ((a & 1) == 2) {
;       x = a * a;
;     }
;     keep(a);
;   }
;   return x;
; }
;
; int main(int argc, char *argv[]) {
;   // f() is inlined here.
;   return f(argc) + 1;
; }
; $ clang -O2 -g shadow.cpp -o shadow.ll -S -emit-llvm

; RUN: llc -filetype=obj -o - %s | llvm-dwarfdump - | FileCheck %s --implicit-check-not=DW_TAG

; CHECK: DW_TAG_compile_unit
; CHECK:  DW_TAG_subprogram
; CHECK:    DW_AT_linkage_name ("_Z1fi")
; CHECK:    DW_AT_inline       (DW_INL_inlined)

; CHECK: [[ABSTRACT_LB:0x[0-9a-f]+]]: DW_TAG_lexical_block
; Abstract DIE for optimized away x.
; CHECK:      DW_TAG_variable
; CHECK:        DW_AT_name  ("x")

; CHECK: DW_TAG_base_type

; CHECK: DW_TAG_subprogram
; CHECK:  DW_AT_name      ("main")
; CHECK:  DW_TAG_inlined_subroutine
; CHECK:    DW_AT_abstract_origin {{.*}}"_Z1fi"
; Concrete block linked to the abstract block with optimized away x should be emitted.
; CHECK:    DW_TAG_lexical_block
; CHECK:      DW_AT_low_pc
; CHECK:      DW_AT_high_pc
; CHECK:      DW_AT_abstract_origin ([[ABSTRACT_LB]])

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

define i32 @main() !dbg !3 {
entry:
  ret i32 0, !dbg !6
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 24.0.0git", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "shadow.cpp", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 24, type: !4, scopeLine: 24, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !5)
!4 = distinct !DISubroutineType(types: !5)
!5 = !{}
!6 = !DILocation(line: 19, column: 5, scope: !7, inlinedAt: !13)
!7 = distinct !DILexicalBlock(scope: !8, file: !1, line: 13, column: 3)
!8 = distinct !DISubprogram(name: "f", linkageName: "_Z1fi", scope: !1, file: !1, line: 10, type: !9, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!9 = distinct !DISubroutineType(types: !5)
!10 = !{!11}
!11 = !DILocalVariable(name: "x", scope: !7, file: !1, line: 15, type: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = distinct !DILocation(line: 26, column: 10, scope: !3)
