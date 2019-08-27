; RUN: llc -mtriple=x86_64-unknown-unknown %s -o - -stop-after=livedebugvalues | FileCheck %s
;
; In the simple loop below, the location of the variable "toast" is %bar in
; the entry block, then set to constant zero at the end of the loop. We cannot
; know the location of "toast" at the start of the %loop block. Test that no
; location is given until after the call to @booler.
;
; CHECK: ![[VARNUM:[0-9]+]] = !DILocalVariable(name: "toast"
;
; CHECK-LABEL: bb.1.loop
; CHECK-NOT:   DBG_VALUE
; CHECK-LABEL: CALL64pcrel32 @booler
; CHECK:       DBG_VALUE 0, $noreg, ![[VARNUM]]

declare i1 @booler()
declare void @escape(i32)
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
