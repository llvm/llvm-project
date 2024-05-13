; RUN: opt < %s -passes=lcssa -S | FileCheck %s

; This test ensures that LCSSA does not insert dbg.value intrinsics using
; insertDebugValuesForPHIs() which effectively cause assignments to be
; re-ordered.
; See PR48206 for more information.

define dso_local i32 @_Z5lcssab(i1 zeroext %S2) {
entry:
  br label %loop.interior

loop.interior:                                    ; preds = %post.if, %entry
  br i1 %S2, label %if.true, label %if.false

if.true:                                          ; preds = %loop.interior
  %X1 = add i32 0, 0
  br label %post.if

if.false:                                         ; preds = %loop.interior
  %X2 = add i32 0, 1
  br label %post.if

post.if:                                          ; preds = %if.false, %if.true
  %X3 = phi i32 [ %X1, %if.true ], [ %X2, %if.false ], !dbg !21
  call void @llvm.dbg.value(metadata i32 %X3, metadata !9, metadata !DIExpression()), !dbg !21
  %Y1 = add i32 4, %X3, !dbg !22
  call void @llvm.dbg.value(metadata i32 %Y1, metadata !9, metadata !DIExpression()), !dbg !22
  br i1 %S2, label %loop.exit, label %loop.interior, !dbg !23

loop.exit:                                        ; preds = %post.if
; CHECK: loop.exit:
; CHECK-NEXT: %X3.lcssa = phi i32
; CHECK-NOT: call void @llvm.dbg.value
  %X4 = add i32 3, %X3
  ret i32 %X4
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify and Author", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "./testcase.ll", directory: "/")
!2 = !{}
!3 = !{i32 11}
!4 = !{i32 5}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "_Z5lcssab", linkageName: "_Z5lcssab", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !{!9})
!7 = !DISubroutineType(types: !2)
!9 = !DILocalVariable(name: "var", scope: !6, file: !1, line: 3, type: !10)
!10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!21 = !DILocation(line: 7, column: 1, scope: !6)
!22 = !DILocation(line: 8, column: 1, scope: !6)
!23 = !DILocation(line: 9, column: 1, scope: !6)

