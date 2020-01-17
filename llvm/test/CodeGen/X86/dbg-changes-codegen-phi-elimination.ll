; RUN: llc -O0 %s -mtriple=x86_64 -stop-after=phi-node-elimination -verify-machineinstrs -o - | FileCheck %s

; Test that the DBG_VALUE must be after PHI and label, the PHI is converted to
; a copy and lower after Label. The DBG_VALUE should not impact PHI lowering.
; before:
;    phi
;    dbg_value
;    label
; after:
;    LABEL
;    COPY
;    DBG_VALUE
; Fix the issue: https://bugs.llvm.org/show_bug.cgi?id=43859

define i32 @func(i32 %a0) personality void ()* @personality !dbg !9 {
bb1:
  %v = icmp slt i32 %a0, 0
  br i1 %v, label %bb2, label %bb3

bb2:
  br label %bb3

bb3:
; CHECK: bb.2.bb3:
; CHECK: EH_LABEL
; CHECK-NEXT: %0:gr32 = COPY %7
; CHECK-NEXT: DBG_VALUE %0
  %0 = phi i32 [ 2, %bb2 ], [ 1, %bb1 ]
  call void @llvm.dbg.value(metadata i32 %0, metadata !8, metadata !DIExpression()), !dbg !13
  invoke void @foo() to label %end unwind label %lpad

end:
  ret i32 %0

lpad:
  %1 = landingpad { i8*, i32 }
           cleanup
  unreachable
}

declare void @foo()

declare void @personality()

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata)


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "a.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, !"min_enum_size", i32 4}
!7 = !{!"clang"}
!8 = !DILocalVariable(name: "c", scope: !9, file: !1, line: 3, type: !12)
!9 = distinct !DISubprogram(name: "func", scope: !1, file: !1, line: 1, type: !10, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12}
!12 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 3, column: 13, scope: !9)
