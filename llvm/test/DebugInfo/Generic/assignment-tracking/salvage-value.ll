; RUN: opt %s -S -o - -passes=instcombine -experimental-assignment-tracking \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"

;; Hand-written (the debug info doesn't necessarily make sense and isn't fully
;; formed). Test salvaging a dbg.assign value and address. Checks and comments
;; are inline.

define dso_local void @fun(i32  %x, i32 %y, ptr %p) !dbg !7 {
entry:
  %add = add nsw i32 %x, 1, !dbg !22
  call void @llvm.dbg.assign(metadata i32 %add, metadata !14, metadata !DIExpression(), metadata !28, metadata ptr %p, metadata !DIExpression()), !dbg !16
;; %add is salvaged.
; CHECK: call void @llvm.dbg.assign(metadata i32 %x,{{.+}}metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value),{{.+}}, metadata ptr %p, metadata !DIExpression())

 %add1 = add nsw i32 %x, %y, !dbg !29
 call void @llvm.dbg.assign(metadata i32 %add1, metadata !14, metadata !DIExpression(), metadata !31, metadata ptr %p, metadata !DIExpression()), !dbg !16
;; %add1 is not salvaged as it requires two values and DIArgList is
;; not (yet) supported for dbg.assigns.
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 undef,{{.+}}, metadata !DIExpression(),{{.+}}, metadata ptr %p, metadata !DIExpression())

  %arrayidx0 = getelementptr inbounds i32, ptr %p, i32 0
  call void @llvm.dbg.assign(metadata i32 %x, metadata !14, metadata !DIExpression(), metadata !17, metadata ptr %arrayidx0, metadata !DIExpression()), !dbg !16
;; %arrayidx0 is salvaged (zero offset, so the gep is just replaced with %p).
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 %x,{{.+}}, metadata !DIExpression(),{{.+}}, metadata ptr %p, metadata !DIExpression())

  %arrayidx1 = getelementptr inbounds i32, ptr %p, i32 1
  call void @llvm.dbg.assign(metadata i32 %x, metadata !14, metadata !DIExpression(), metadata !18, metadata ptr %arrayidx1, metadata !DIExpression()), !dbg !16
;; %arrayidx1 is salvaged.
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 %x,{{.+}}, metadata !DIExpression(),{{.+}}, metadata ptr %p, metadata !DIExpression(DW_OP_plus_uconst, 4))

  %arrayidx2 = getelementptr inbounds i32, ptr %p, i32 %x
  call void @llvm.dbg.assign(metadata i32 %x, metadata !14, metadata !DIExpression(), metadata !19, metadata ptr %arrayidx2, metadata !DIExpression()), !dbg !16
;; Variadic DIExpressions for dbg.assign address component is not supported -
;; set undef.
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 %x,{{.+}}, metadata !DIExpression(),{{.+}}, metadata ptr undef, metadata !DIExpression())

  ret void
}

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{!"clang version 14.0.0"}
!7 = distinct !DISubprogram(name: "fun", linkageName: "fun", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !13, !14}
!12 = !DILocalVariable(name: "x", arg: 1, scope: !7, file: !1, line: 2, type: !10)
!13 = !DILocalVariable(name: "y", arg: 2, scope: !7, file: !1, line: 2, type: !10)
!14 = !DILocalVariable(name: "Local", scope: !7, file: !1, line: 3, type: !10)
!16 = !DILocation(line: 0, scope: !7)
!17 = distinct !DIAssignID()
!18 = distinct !DIAssignID()
!19 = distinct !DIAssignID()
!21 = !DILocation(line: 3, column: 3, scope: !7)
!22 = !DILocation(line: 3, column: 17, scope: !7)
!28 = distinct !DIAssignID()
!29 = !DILocation(line: 4, column: 13, scope: !7)
!31 = distinct !DIAssignID()
