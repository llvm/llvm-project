; RUN: llc -mtriple=aarch64 -stop-after=finalize-isel < %s | FileCheck %s

; Refining the extload to a zextload leaves the truncate dead, and the truncate
; is where the location of %a had been parked. Its debug values have to move to
; the replacement, or "a" loses its location.

; The metadata block precedes the function body in MIR, so bind the variables
; first and then require both to have a register location.
; CHECK-DAG: ![[A:[0-9]+]] = !DILocalVariable(name: "a"
; CHECK-DAG: ![[B:[0-9]+]] = !DILocalVariable(name: "b"
; CHECK-LABEL: name: f
; CHECK-DAG: DBG_VALUE %[[R:[0-9]+]], $noreg, ![[A]], !DIExpression()
; CHECK-DAG: DBG_VALUE %[[R]], $noreg, ![[B]], !DIExpression()

declare void @sink(i8, i64)

define void @f(ptr %p) nounwind !dbg !5 {
entry:
  %a = load i8, ptr %p, align 1, !dbg !9
    #dbg_value(i8 %a, !10, !DIExpression(), !9)
  %b = zext i8 %a to i64, !dbg !9
    #dbg_value(i64 %b, !12, !DIExpression(), !9)
  call void @sink(i8 %a, i64 %b), !dbg !9
  ret void, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{!10, !12}
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DILocalVariable(name: "a", scope: !5, file: !1, line: 2, type: !11)
!11 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned_char)
!12 = !DILocalVariable(name: "b", scope: !5, file: !1, line: 3, type: !13)
!13 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_unsigned)
