; RUN: opt < %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s

; Check that the debug location for the hoisted store for "ret = 0" is a
; line-0 location.
;
; int foo(int x) {
;   int ret = 1;
;   if (x)
;     ret = 0;
;   return ret;
; }
;
; CHECK: store i32 1,{{.+}}!dbg ![[DLOC1:[0-9]+]]
; CHECK: icmp ne {{.+}}!dbg ![[DLOC2:[0-9]+]]
; CHECK: [[VREG:%[^ ]+]] = select
; CHECK: store i32 [[VREG]],{{.*}} !dbg [[storeLoc:![0-9]+]]
; CHECK: ret {{.+}}!dbg ![[DLOC3:[0-9]+]]
; CHECK: ![[DLOC1]] = !DILocation(line: 2
; CHECK: ![[DLOC2]] = !DILocation(line: 3
; CHECK: [[storeLoc]] = !DILocation(line: 0
; CHECK: ![[DLOC3]] = !DILocation(line: 5

target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define i32 @foo(i32) !dbg !6 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  store i32 1, i32* %3, align 4, !dbg !14
  %4 = load i32, i32* %2, align 4, !dbg !15
  %5 = icmp ne i32 %4, 0, !dbg !15
  br i1 %5, label %6, label %7, !dbg !17

; <label>:6:                                      ; preds = %1
  store i32 0, i32* %3, align 4, !dbg !18
  br label %7, !dbg !19

; <label>:7:                                      ; preds = %6, %1
  %8 = load i32, i32* %3, align 4, !dbg !20
  ret i32 %8, !dbg !21
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1)
!1 = !DIFile(filename: "foo.c", directory: "b/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocalVariable(name: "x", arg: 1, scope: !6, file: !1, line: 1, type: !9)
!11 = !DIExpression()
!12 = !DILocation(line: 1, column: 13, scope: !6)
!13 = !DILocalVariable(name: "ret", scope: !6, file: !1, line: 2, type: !9)
!14 = !DILocation(line: 2, column: 7, scope: !6)
!15 = !DILocation(line: 3, column: 7, scope: !16)
!16 = distinct !DILexicalBlock(scope: !6, file: !1, line: 3, column: 7)
!17 = !DILocation(line: 3, column: 7, scope: !6)
!18 = !DILocation(line: 4, column: 9, scope: !16)
!19 = !DILocation(line: 4, column: 5, scope: !16)
!20 = !DILocation(line: 5, column: 10, scope: !6)
!21 = !DILocation(line: 5, column: 3, scope: !6)
