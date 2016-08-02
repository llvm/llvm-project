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

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = common global i32 0, align 4
@y = common global i32 0, align 4

define i32 @main() !dbg !4 {
; CHECK: .loc 1 {{[89]}}
; CHECK-NOT: .loc
; CHECK: movl $1

entry:
  %0 = load volatile i32, i32* @x, align 4, !dbg !16, !tbaa !19
  %mul.i = mul nsw i32 %0, 3, !dbg !23
  %cmp.i = icmp slt i32 %mul.i, 14, !dbg !24
  %..i = select i1 %cmp.i, i32 1, i32 2, !dbg !25
  store volatile i32 %..i, i32* @x, align 4, !dbg !27, !tbaa !19
  %1 = load volatile i32, i32* @x, align 4, !dbg !28, !tbaa !19
  %tobool = icmp ne i32 %1, 0, !dbg !28
  br i1 %tobool, label %select.end, label %select.mid

select.mid:                                       ; preds = %entry
  br label %select.end

select.end:                                       ; preds = %entry, %select.mid
  %cond = phi i32 [ 1, %entry ], [ 2, %select.mid ]
  store volatile i32 %cond, i32* @x, align 4, !dbg !29, !tbaa !19
  ret i32 0, !dbg !30
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !9, imports: !2)
!1 = !DIFile(filename: "inline-break.c", directory: "/build/dir")
!2 = !{}
!3 = !{!4, !8}
!4 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 7, type: !5, isLocal: false, isDefinition: true, scopeLine: 7, isOptimized: true, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = distinct !DISubprogram(name: "f1", scope: !1, file: !1, line: 3, type: !5, isLocal: true, isDefinition: true, scopeLine: 3, isOptimized: true, variables: !2)
!9 = !{!10, !12}
!10 = !DIGlobalVariable(name: "x", scope: !0, file: !1, line: 1, type: !11, isLocal: false, isDefinition: true, variable: i32* @x)
!11 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !7)
!12 = !DIGlobalVariable(name: "y", scope: !0, file: !1, line: 2, type: !7, isLocal: false, isDefinition: true, variable: i32* @y)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !DILocation(line: 4, column: 9, scope: !17, inlinedAt: !18)
!17 = distinct !DILexicalBlock(scope: !8, file: !1, line: 4, column: 9)
!18 = distinct !DILocation(line: 8, column: 9, scope: !4)
!19 = !{!20, !20, i64 0}
!20 = !{!"int", !21, i64 0}
!21 = !{!"omnipotent char", !22, i64 0}
!22 = !{!"Simple C/C++ TBAA"}
!23 = !DILocation(line: 4, column: 11, scope: !17, inlinedAt: !18)
!24 = !DILocation(line: 4, column: 15, scope: !17, inlinedAt: !18)
!25 = !DILocation(line: 4, column: 21, scope: !26, inlinedAt: !18)
!26 = !DILexicalBlockFile(scope: !17, file: !1, discriminator: 1)
!27 = !DILocation(line: 8, column: 7, scope: !4)
!28 = !DILocation(line: 9, column: 9, scope: !4)
!29 = !DILocation(line: 9, column: 7, scope: !4)
!30 = !DILocation(line: 10, column: 1, scope: !4)
