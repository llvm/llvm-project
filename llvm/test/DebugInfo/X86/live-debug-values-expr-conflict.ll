; RUN: llc %s -mtriple=x86_64-unknown-unknown -o - -stop-after=livedebugvalues | FileCheck %s
;
; In the C below, 'baz' is re-assigned with a value that gets salvaged, making
; it's dbg.value base itself on 'bar', but with a complex expression.
; LiveDebugValues should recognize that these are different locations, and not
; propagate a location for 'baz' into the return block.
;
; void escape1(int bees);
; void escape2(int bees);
; 
; int foo(int bar) {
;   int baz = bar;
;   if (baz == 12) {
;     escape1(bar);
;   } else {
;     baz += 1;
;     escape2(bar);
;   }
; 
;   return bar;
; }
;
; We should get a plain DBG_VALUE in the entry block, a plain one then complex
; one in the block two, and none in block three.
; CHECK:       ![[BAZVAR:[0-9]+]] = !DILocalVariable(name: "baz",
; CHECK-LABEL: bb.0.entry:
; CHECK:       DBG_VALUE {{[0-9a-zA-Z$%_]*}}, $noreg, ![[BAZVAR]],
; CHECK-SAME:     !DIExpression()
; CHECK-LABEL: bb.1.if.then:
; CHECK-LABEL: bb.2.if.else:
; CHECK:       DBG_VALUE {{[0-9a-zA-Z$%_]*}}, $noreg, ![[BAZVAR]],
; CHECK-SAME:     !DIExpression()
; CHECK:       DBG_VALUE {{[0-9a-zA-Z$%_]*}}, $noreg, ![[BAZVAR]],
; CHECK-SAME:     !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)
; CHECK-LABEL: bb.3.if.end:
; CHECK-NOT:   DBG_VALUE

declare void @escape1(i32)
declare void @escape2(i32)
declare void @llvm.dbg.value(metadata, metadata, metadata)

define i32 @foo(i32 returned %bar) !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 %bar, metadata !13, metadata !DIExpression()), !dbg !14
  %cmp = icmp eq i32 %bar, 12, !dbg !14
  br i1 %cmp, label %if.then, label %if.else, !dbg !14

if.then:
  tail call void @escape1(i32 12) #3, !dbg !14
  br label %if.end, !dbg !14

if.else:
  call void @llvm.dbg.value(metadata i32 %bar, metadata !13, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !14
  tail call void @escape2(i32 %bar) #3, !dbg !14
  br label %if.end

if.end:
  ret i32 %bar, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "exprconflict.c", directory: "/home/jmorse")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!13}
!13 = !DILocalVariable(name: "baz", scope: !7, file: !1, line: 6, type: !10)
!14 = !DILocation(line: 1, scope: !7)
