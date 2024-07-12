;; Test that we can write in the old debug info format.
; RUN: opt --passes=verify -S --write-experimental-debuginfo=false < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,OLDDBG --implicit-check-not=llvm.dbg --implicit-check-not=#dbg

;; Test that we can write in the new debug info format...
; RUN: opt --passes=verify -S --write-experimental-debuginfo=true < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NEWDBG --implicit-check-not=llvm.dbg --implicit-check-not=#dbg

;; ...and then read the new format and write the old format.
; RUN: opt --passes=verify -S --write-experimental-debuginfo=true < %s \
; RUN:   | opt --passes=verify -S --write-experimental-debuginfo=false \
; RUN:   | FileCheck %s --check-prefixes=CHECK,OLDDBG  --implicit-check-not=llvm.dbg --implicit-check-not=#dbg

;; Test also that the new flag is independent of the flag that enables use of
;; these non-instruction debug info during LLVM passes.
; RUN: opt --passes=verify -S --try-experimental-debuginfo-iterators --write-experimental-debuginfo=false < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,OLDDBG --implicit-check-not=llvm.dbg --implicit-check-not=#dbg
; RUN: opt --passes=verify -S --try-experimental-debuginfo-iterators --write-experimental-debuginfo=true < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NEWDBG --implicit-check-not=llvm.dbg --implicit-check-not=#dbg

;; Test that the preserving flag overrides the write flag.
; RUN: opt --passes=verify -S --preserve-input-debuginfo-format=true --write-experimental-debuginfo=true < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,OLDDBG  --implicit-check-not=llvm.dbg --implicit-check-not=#dbg

; RUN: opt --passes=verify -S --write-experimental-debuginfo=true < %s \
; RUN:   | opt --passes=verify -S --preserve-input-debuginfo-format=true --write-experimental-debuginfo=false \
; RUN:   | FileCheck %s --check-prefixes=CHECK,NEWDBG  --implicit-check-not=llvm.dbg --implicit-check-not=#dbg

; CHECK: @f(i32 %[[VAL_A:[0-9a-zA-Z]+]])
; CHECK-NEXT: entry:
; OLDDBG-NEXT: call void @llvm.dbg.value(metadata i32 %[[VAL_A]], metadata ![[VAR_A:[0-9]+]], metadata !DIExpression()), !dbg ![[LOC_1:[0-9]+]]
; NEWDBG-NEXT: {{^}}    #dbg_value(i32 %[[VAL_A]], ![[VAR_A:[0-9]+]], !DIExpression(), ![[LOC_1:[0-9]+]])
; CHECK-NEXT: {{^}}  %[[VAL_B:[0-9a-zA-Z]+]] = alloca
; OLDDBG-NEXT: call void @llvm.dbg.declare(metadata ptr %[[VAL_B]], metadata ![[VAR_B:[0-9]+]], metadata !DIExpression()), !dbg ![[LOC_2:[0-9]+]]
; NEWDBG-NEXT: {{^}}    #dbg_declare(ptr %[[VAL_B]], ![[VAR_B:[0-9]+]], !DIExpression(), ![[LOC_2:[0-9]+]])
; CHECK-NEXT: {{^}}  %[[VAL_ADD:[0-9a-zA-Z]+]] = add i32 %[[VAL_A]], 5
; OLDDBG-NEXT: call void @llvm.dbg.value(metadata !DIArgList(i32 %[[VAL_A]], i32 %[[VAL_ADD]]), metadata ![[VAR_A]], metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus)), !dbg ![[LOC_3:[0-9]+]]
; NEWDBG-NEXT: {{^}}    #dbg_value(!DIArgList(i32 %[[VAL_A]], i32 %[[VAL_ADD]]), ![[VAR_A]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus), ![[LOC_3:[0-9]+]])
; OLDDBG-NEXT: call void @llvm.dbg.label(metadata ![[LABEL_ID:[0-9]+]]), !dbg ![[LOC_3]]
; NEWDBG-NEXT: {{^}}    #dbg_label(![[LABEL_ID:[0-9]+]], ![[LOC_3]])
; CHECK-NEXT: {{^}}  store i32 %[[VAL_ADD]]{{.+}}, !DIAssignID ![[ASSIGNID:[0-9]+]]
; OLDDBG-NEXT: call void @llvm.dbg.assign(metadata i32 %[[VAL_ADD]], metadata ![[VAR_B]], metadata !DIExpression(), metadata ![[ASSIGNID]], metadata ptr %[[VAL_B]], metadata !DIExpression()), !dbg ![[LOC_4:[0-9]+]]
; NEWDBG-NEXT: {{^}}    #dbg_assign(i32 %[[VAL_ADD]], ![[VAR_B]], !DIExpression(), ![[ASSIGNID]], ptr %[[VAL_B]], !DIExpression(), ![[LOC_4:[0-9]+]])
; CHECK-NEXT: {{^}}  ret i32

; OLDDBG-DAG: declare void @llvm.dbg.value
; OLDDBG-DAG: declare void @llvm.dbg.declare
; OLDDBG-DAG: declare void @llvm.dbg.assign
; OLDDBG-DAG: declare void @llvm.dbg.label

; CHECK-DAG: llvm.dbg.cu
; CHECK-DAG: ![[VAR_A]] = !DILocalVariable(name: "a"
; CHECK-DAG: ![[VAR_B]] = !DILocalVariable(name: "b"
; CHECK-DAG: ![[LOC_1]] = !DILocation(line: 3, column: 15
; CHECK-DAG: ![[LOC_2]] = !DILocation(line: 3, column: 20
; CHECK-DAG: ![[LOC_3]] = !DILocation(line: 3, column: 25
; CHECK-DAG: ![[LOC_4]] = !DILocation(line: 3, column: 30
; CHECK-DAG: ![[LABEL_ID]] = !DILabel(

define dso_local i32 @f(i32 %a) !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 %a, metadata !20, metadata !DIExpression()), !dbg !30
  %b = alloca i32, !dbg !30, !DIAssignID !40
  call void @llvm.dbg.declare(metadata ptr %b, metadata !21, metadata !DIExpression()), !dbg !31
  %add = add i32 %a, 5, !dbg !31
  call void @llvm.dbg.value(metadata !DIArgList(i32 %a, i32 %add), metadata !20, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus)), !dbg !32
  call void @llvm.dbg.label(metadata !50), !dbg !32
  store i32 %add, ptr %b, !dbg !32, !DIAssignID !40
  call void @llvm.dbg.assign(metadata i32 %add, metadata !21, metadata !DIExpression(), metadata !40, metadata ptr %b, metadata !DIExpression()), !dbg !33
  ret i32 %add, !dbg !33

}

declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)
declare void @llvm.dbg.label(metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 18.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "print.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 18.0.0"}
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !13)
!8 = !DISubroutineType(types: !9)
!9 = !{!12, !12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{!20, !21}
!20 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 3, type: !12)
!21 = !DILocalVariable(name: "b", scope: !7, file: !1, line: 3, type: !12)
!30 = !DILocation(line: 3, column: 15, scope: !7)
!31 = !DILocation(line: 3, column: 20, scope: !7)
!32 = !DILocation(line: 3, column: 25, scope: !7)
!33 = !DILocation(line: 3, column: 30, scope: !7)
!40 = distinct !DIAssignID()
!50 = !DILabel(scope: !7, name: "label", file: !1, line: 3)