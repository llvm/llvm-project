; RUN: opt -S -passes=lcssa < %s | FileCheck %s
; RUN: opt -S -passes=lcssa < %s --try-experimental-debuginfo-iterators | FileCheck %s

; Reproducer for PR39019.
;
; Verify that the llvm.dbg.values are updated to use the PHI nodes inserted by
; LCSSA.

; For the test case @single_exit, we can rewrite all llvm.dbg.value calls
; to use the inserted PHI.

; CHECK-LABEL: @single_exit(

; CHECK-LABEL: inner.body:
; CHECK: %add = add nsw i32 0, 2
; CHECK: call void @llvm.dbg.value(metadata i32 %add, metadata [[VAR:![0-9]+]], metadata !DIExpression())


; CHECK-LABEL: outer.exit:
; CHECK-NEXT: [[PN:%[^ ]*]] = phi i32 [ %add.lcssa, %outer.latch ]
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 [[PN]], metadata [[VAR]], metadata !DIExpression())
; CHECK-NEXT: call void @bar(i32 [[PN]])

; CHECK-LABEL: exit:
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 [[PN]], metadata [[VAR]], metadata !DIExpression())

define void @single_exit()  !dbg !6 {
entry:
  br label %outer.header, !dbg !12

outer.header:                                     ; preds = %outer.latch, %entry
  br label %inner.body, !dbg !12

inner.body:                                       ; preds = %inner.body, %outer.header
  %add = add nsw i32 0, 2, !dbg !12
  call void @llvm.dbg.value(metadata i32 %add, metadata !9, metadata !DIExpression()), !dbg !12
  br i1 false, label %inner.body, label %inner.exit, !dbg !12

inner.exit:                                       ; preds = %inner.body
  br label %outer.latch

outer.latch:                                      ; preds = %inner.exit
  br i1 false, label %outer.header, label %outer.exit, !dbg !12

outer.exit:                                       ; preds = %outer.latch
  call void @llvm.dbg.value(metadata i32 %add, metadata !9, metadata !DIExpression()), !dbg !12
  tail call void @bar(i32 %add), !dbg !12
  br label %exit

exit:                                             ; preds = %outer.exit
  call void @llvm.dbg.value(metadata i32 %add, metadata !9, metadata !DIExpression()), !dbg !12
  ret void, !dbg !12
}

; For the test case @multi_exit, we cannot update the llvm.dbg.value call in exit,
; because LCSSA did not insert a PHI node in %exit, as there is no non-debug
; use.

; CHECK-LABEL: @multi_exit()

; CHECK-LABEL: for.header:
; CHECK-NEXT: %add = add nsw i32 0, 2
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 %add, metadata [[VAR2:![0-9]+]], metadata !DIExpression())

; CHECK-LABEL: for.exit1:
; CHECK-NEXT: [[PN1:%[^ ]*]] = phi i32 [ %add, %for.header ]
; CHECK-NEXT: br label %for.exit1.succ

; CHECK-LABEL: for.exit1.succ:
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 [[PN1]], metadata [[VAR2]], metadata !DIExpression())
; CHECK-NEXT: call void @bar(i32 [[PN1]])

; CHECK-LABEL: for.exit2:
; CHECK-NEXT: [[PN2:%[^ ]*]] = phi i32 [ %add, %for.latch ]
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 [[PN2]], metadata [[VAR2]], metadata !DIExpression())
; CHECK-NEXT: call void @bar(i32 [[PN2]])

; CHECK-LABEL: exit:
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 %add, metadata [[VAR2]], metadata !DIExpression())

define void @multi_exit()  !dbg !13 {
entry:
  br label %for.header, !dbg !14

for.header:                                       ; preds = %for.latch, %entry
  %add = add nsw i32 0, 2, !dbg !14
  call void @llvm.dbg.value(metadata i32 %add, metadata !16, metadata !DIExpression()), !dbg !14
  br i1 false, label %for.latch, label %for.exit1, !dbg !14

for.latch:                                        ; preds = %for.header
  br i1 false, label %for.header, label %for.exit2, !dbg !14

for.exit1:                                        ; preds = %for.header
  br label %for.exit1.succ

for.exit1.succ:                                   ; preds = %for.exit1
  call void @llvm.dbg.value(metadata i32 %add, metadata !16, metadata !DIExpression()), !dbg !14
  tail call void @bar(i32 %add), !dbg !14
  br label %exit

for.exit2:                                        ; preds = %for.latch
  call void @llvm.dbg.value(metadata i32 %add, metadata !16, metadata !DIExpression()), !dbg !14
  tail call void @bar(i32 %add), !dbg !14
  br label %exit

exit:                                             ; preds = %for.exit2, %for.exit1.succ
  call void @llvm.dbg.value(metadata i32 %add, metadata !16, metadata !DIExpression()), !dbg !14
  ret void, !dbg !14
}

; CHECK: [[VAR]] = !DILocalVariable(name: "sum",
; CHECK: [[VAR2]] = !DILocalVariable(name: "sum2",

declare void @bar(i32)

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !2, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 8.0.0"}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 10, type: !7, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!9}
!9 = !DILocalVariable(name: "sum", scope: !10, file: !1, line: 11, type: !11)
!10 = !DILexicalBlockFile(scope: !6, file: !1, discriminator: 0)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocation(line: 0, scope: !10)
!13 = distinct !DISubprogram(name: "multi_exit", scope: !1, file: !1, line: 10, type: !7, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!14 = !DILocation(line: 0, scope: !15)
!15 = !DILexicalBlockFile(scope: !13, file: !1, discriminator: 0)
!16 = !DILocalVariable(name: "sum2", scope: !15, file: !1, line: 11, type: !11)
