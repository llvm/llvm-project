; RUN: llc -filetype=asm -asm-verbose=0 < %s | FileCheck %s

; "1" from line 09 in the snippet below shouldn't be marked with location of "1"
; from line 04.  Instead it will have location inside main() (real location is
; just erased, so it won't be perfectly accurate).

; options: -g -O3
; 01 volatile int x;
; 02 int y;
; 03 static __attribute__((always_inline)) int f1() {
; 04     if (x * 3 < 14) return 1;
; 05     return 2;
; 06 }
; 07 int main() {
; 08     x = f1();
; 09     x = x ? 1 : 2;
; 10 }

source_filename = "test/DebugInfo/X86/inlined-indirect-value.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = common global i32 0, align 4, !dbg !0
@y = common global i32 0, align 4, !dbg !6

define i32 @main() !dbg !12 {
; CHECK: .loc 1 {{[89]}}
; CHECK-NOT: .loc
; CHECK: movl $1

entry:
  %0 = load volatile i32, ptr @x, align 4, !dbg !15, !tbaa !19
  %mul.i = mul nsw i32 %0, 3, !dbg !23
  %cmp.i = icmp slt i32 %mul.i, 14, !dbg !24
  %..i = select i1 %cmp.i, i32 1, i32 2, !dbg !25
  store volatile i32 %..i, ptr @x, align 4, !dbg !27, !tbaa !19
  %1 = load volatile i32, ptr @x, align 4, !dbg !28, !tbaa !19
  %tobool = icmp ne i32 %1, 0, !dbg !28
  br i1 %tobool, label %select.end, label %select.mid

select.mid:                                       ; preds = %entry
  br label %select.end

select.end:                                       ; preds = %select.mid, %entry
  %cond = phi i32 [ 1, %entry ], [ 2, %select.mid ]
  store volatile i32 %cond, ptr @x, align 4, !dbg !29, !tbaa !19
  ret i32 0, !dbg !30
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !9, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !5, imports: !4)
!3 = !DIFile(filename: "inline-break.c", directory: "/build/dir")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = !DIGlobalVariable(name: "y", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !8)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 7, type: !13, isLocal: false, isDefinition: true, scopeLine: 7, isOptimized: true, unit: !2, retainedNodes: !4)
!13 = !DISubroutineType(types: !14)
!14 = !{!8}
!15 = !DILocation(line: 4, column: 9, scope: !16, inlinedAt: !18)
!16 = distinct !DILexicalBlock(scope: !17, file: !3, line: 4, column: 9)
!17 = distinct !DISubprogram(name: "f1", scope: !3, file: !3, line: 3, type: !13, isLocal: true, isDefinition: true, scopeLine: 3, isOptimized: true, unit: !2, retainedNodes: !4)
!18 = distinct !DILocation(line: 8, column: 9, scope: !12)
!19 = !{!20, !20, i64 0}
!20 = !{!"int", !21, i64 0}
!21 = !{!"omnipotent char", !22, i64 0}
!22 = !{!"Simple C/C++ TBAA"}
!23 = !DILocation(line: 4, column: 11, scope: !16, inlinedAt: !18)
!24 = !DILocation(line: 4, column: 15, scope: !16, inlinedAt: !18)
!25 = !DILocation(line: 4, column: 21, scope: !26, inlinedAt: !18)
!26 = !DILexicalBlockFile(scope: !16, file: !3, discriminator: 1)
!27 = !DILocation(line: 8, column: 7, scope: !12)
!28 = !DILocation(line: 9, column: 9, scope: !12)
!29 = !DILocation(line: 9, column: 7, scope: !12)
!30 = !DILocation(line: 10, column: 1, scope: !12)

