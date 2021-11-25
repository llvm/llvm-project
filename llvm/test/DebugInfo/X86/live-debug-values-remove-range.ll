; RUN: llc -mtriple=x86_64-unknown-unknown %s -o - -stop-after=livedebugvalues -experimental-debug-variable-locations=true | FileCheck %s
;
; In the simple loop below, the location of the variable "toast" is %bar in
; the entry block, then set to constant zero at the end of the loop. We cannot
; know the location of "toast" at the start of the %loop block. Test that no
; location is given until after the call to @booler.
;
; Second function @baz added with an even tighter loop -- this tests different
; code-paths through LiveDebugValues. Any blocks with an incoming backedge need
; reconsideration after the parent of the backedge has had its OutLocs
; initialized, even if OutLocs hasn't changed.
;
; Third function @quux tests that we don't delete too many variable locations.
; A variable that is live across the body of the loop should maintain its
; location across that loop, and not be invalidated.
;
; CHECK: ![[FOOVARNUM:[0-9]+]] = !DILocalVariable(name: "toast"
; CHECK: ![[BAZVARNUM:[0-9]+]] = !DILocalVariable(name: "crumpets"
; CHECK: ![[QUUXVARNUM:[0-9]+]] = !DILocalVariable(name: "teacake"
;
; foo tests
; CHECK-LABEL: bb.1.loop
; CHECK-NOT:   DBG_VALUE
; CHECK-LABEL: CALL64pcrel32 @booler
; CHECK:       DBG_VALUE 0, $noreg, ![[FOOVARNUM]]
;
; baz tests
; CHECK-LABEL: name: baz
; CHECK-LABEL: bb.1.loop
; CHECK-NOT:   DBG_VALUE
; CHECK-LABEL: CALL64pcrel32 @booler
; CHECK:       DBG_VALUE 0, $noreg, ![[BAZVARNUM]]
;
; quux tests -- the variable arrives in $edi, should get a non-undef location
; before the loop, and its position re-stated in each block.
; CHECK-LABEL: name: quux
; CHECK:       DBG_VALUE $edi, $noreg, ![[QUUXVARNUM]]
; CHECK-LABEL: bb.1.loop
; CHECK:       DBG_VALUE $ebx, $noreg, ![[QUUXVARNUM]]
; CHECK-NOT:   DBG_VALUE $noreg
; CHECK-LABEL: bb.2.exit
; CHECK:       DBG_VALUE $ebx, $noreg, ![[QUUXVARNUM]]
; CHECK-NOT:   DBG_VALUE $noreg

declare dso_local i1 @booler()
declare dso_local void @escape(i32)
declare void @llvm.dbg.value(metadata, metadata, metadata)
@glob = global i32 0

define i32 @foo(i32 %bar) !dbg !4 {
entry:
  call void @llvm.dbg.value(metadata i32 %bar, metadata !3, metadata !DIExpression()), !dbg !6
  br label %loop
loop:
  call void @escape(i32 %bar)
  %retval = call i1 @booler(), !dbg !6
  call void @llvm.dbg.value(metadata i32 0, metadata !3, metadata !DIExpression()), !dbg !6
  br i1 %retval, label %loop2, label %exit
loop2:
  store i32 %bar, i32 *@glob
  br label %loop
exit:
  ret i32 %bar
}

define i32 @baz(i32 %bar) !dbg !104 {
entry:
  call void @llvm.dbg.value(metadata i32 %bar, metadata !103, metadata !DIExpression()), !dbg !106
  br label %loop
loop:
  call void @escape(i32 %bar)
  %retval = call i1 @booler(), !dbg !106
  call void @llvm.dbg.value(metadata i32 0, metadata !103, metadata !DIExpression()), !dbg !106
  br i1 %retval, label %loop, label %exit
exit:
  ret i32 %bar
}

define i32 @quux(i32 %bar) !dbg !204 {
entry:
  ; %bar will be placed in a nonvolatile or spill location for the loop,
  ; before being returned later.
  call void @llvm.dbg.value(metadata i32 %bar, metadata !203, metadata !DIExpression()), !dbg !206
  br label %loop
loop:
  %retval = call i1 @booler(), !dbg !206
  br i1 %retval, label %loop, label %exit
exit:
  ret i32 %bar
}

!llvm.module.flags = !{!0, !100}
!llvm.dbg.cu = !{!1}

!100 = !{i32 2, !"Dwarf Version", i32 4}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "beards", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!2 = !DIFile(filename: "bees.cpp", directory: ".")
!3 = !DILocalVariable(name: "toast", scope: !4, file: !2, line: 1, type: !16)
!4 = distinct !DISubprogram(name: "nope", scope: !2, file: !2, line: 1, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !13, type: !14, isDefinition: true)
!6 = !DILocation(line: 1, scope: !4)
!13 = !{!3}
!14 = !DISubroutineType(types: !15)
!15 = !{!16}
!16 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!103 = !DILocalVariable(name: "crumpets", scope: !104, file: !2, line: 1, type: !16)
!104 = distinct !DISubprogram(name: "ribbit", scope: !2, file: !2, line: 1, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !113, type: !14, isDefinition: true)
!106 = !DILocation(line: 1, scope: !104)
!113 = !{!103}
!203 = !DILocalVariable(name: "teacake", scope: !204, file: !2, line: 1, type: !16)
!204 = distinct !DISubprogram(name: "toad", scope: !2, file: !2, line: 1, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !113, type: !14, isDefinition: true)
!206 = !DILocation(line: 1, scope: !204)
!213 = !{!203}
