; RUN: opt -passes=simplifycfg -S < %s | FileCheck %s
;
; Simplified from the following code:
; int foo() {
;   if(c) { return a; } else { return b; }
; }
define i32 @foo(i32 %c, i32 %a, i32 %b) !dbg !4 {
; CHECK: define i32 @foo({{.*}}) !dbg [[FOO_SUBPROGRAM:![0-9]+]]
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TOBOOL:%.*]] = icmp ne i32 [[C:%.*]], 0, !dbg [[DBG_CMP:![0-9]+]]
; CHECK-NEXT:    [[A_B:%.*]] = select i1 [[TOBOOL]], i32 [[A:%.*]], i32 [[B:%.*]]
; CHECK-NEXT:    ret i32 [[A_B]], !dbg [[DBG_RET:![0-9]+]]
; CHECK: [[DBG_CMP]] = !DILocation(line: 2, column: 1, scope: [[FOO_SUBPROGRAM]])
; CHECK: [[DBG_RET]] = !DILocation(line: 4, scope: [[FOO_SUBPROGRAM]])
entry:
  %tobool = icmp ne i32 %c, 0, !dbg !7
  br i1 %tobool, label %cond.true, label %cond.false, !dbg !8

cond.true:                                        ; preds = %entry
  ret i32 %a, !dbg !9

cond.false:                                       ; preds = %entry
  ret i32 %b, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 16.0.0)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !6)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 2, column: 1, scope: !4)
!8 = !DILocation(line: 3, column: 1, scope: !4)
!9 = !DILocation(line: 4, column: 1, scope: !4)
!10 = !DILocation(line: 4, column: 2, scope: !4)
