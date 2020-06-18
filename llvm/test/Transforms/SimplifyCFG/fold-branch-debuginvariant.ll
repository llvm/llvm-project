; RUN: opt < %s -simplifycfg -S | FileCheck %s
; RUN: opt < %s -strip-debug -simplifycfg -S | FileCheck %s

; Verify that the and.2 instruction is eliminated even in the presence of a
; preceding debug intrinsic.

; CHECK-LABEL: bb1:
; CHECK: and i1 false, false
; CHECK-NOT: and i1 false, false

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse nounwind
define void @foo() local_unnamed_addr #0 !dbg !4 {
bb1:
  %and.1 = and i1 false, false
  %cmp = icmp eq i16 0, 0
  br i1 %cmp, label %bb2, label %bb3

bb2:                                              ; preds = %bb1
  call void @llvm.dbg.value(metadata i16 0, metadata !8, metadata !DIExpression()), !dbg !9
  %and.2 = and i1 false, false
  br label %bb3

bb3:                                              ; preds = %bb2, %bb1
  ret void
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { norecurse nounwind }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "Foo", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2)
!1 = !DIFile(filename: "foo.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 10, type: !5, isLocal: false, isDefinition: true, scopeLine: 10, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DILocalVariable(name: "p_1", arg: 1, scope: !4, line: 4, type: !7)
!9 = distinct !DILocation(line: 11, column: 3, scope: !4)
