; REQUIRES: x86-registered-target
; RUN: llc -start-before=codegenprepare -stop-after=codegenprepare \
; RUN:   -mtriple=x86_64-unknown-unknown %s -o - \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg."

;; Test with RemoveDIs non-intrinsic debug-info too.
; RUN: llc -start-before=codegenprepare -stop-after=codegenprepare \
; RUN:   -mtriple=x86_64-unknown-unknown %s -o - --try-experimental-debuginfo-iterators \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg."

;; Check that when CodeGenPrepare moves an address computation to a block it's
;; used in its dbg.assign uses are updated.
;;
;; Based on llvm/test/DebugInfo/X86/codegenprepare-addrsink.ll

define dso_local i8 @foo(ptr %p, i32 %cond) !dbg !7 {
entry:
  %casted = bitcast ptr %p to ptr
  %arith = getelementptr i8, ptr %casted, i32 3
  %load1 = load i8, ptr %arith
  %cmpresult = icmp eq i32 %cond, 0
  br i1 %cmpresult, label %next, label %ret

next:
; Address calcs should be duplicated into this block. One dbg.value should be
; updated, and the other should not.
; CHECK-LABEL: next:
; CHECK:       %[[CASTVAR:[0-9a-zA-Z]+]] = bitcast ptr %p to ptr
; CHECK-NEXT:  dbg.assign(metadata ptr %arith, metadata ![[DIVAR:[0-9]+]],
; CHECK-NEXT:  %[[GEPVAR:[0-9a-zA-Z]+]] = getelementptr i8, ptr %[[CASTVAR]], i64 3
; CHECK-NEXT:  %loaded = load i8, ptr %[[GEPVAR]]
; CHECK-NEXT:  dbg.assign(metadata ptr %[[GEPVAR]], metadata ![[DIVAR]],
  call void @llvm.dbg.assign(metadata ptr %arith, metadata !12, metadata !DIExpression(), metadata !21, metadata ptr undef, metadata !DIExpression()), !dbg !14
  %loaded = load i8, ptr %arith
  call void @llvm.dbg.assign(metadata ptr %arith, metadata !12, metadata !DIExpression(), metadata !21, metadata ptr undef, metadata !DIExpression()), !dbg !14
  ret i8 %loaded

ret:
  ret i8 0
}

; CHECK: ![[DIVAR]] = !DILocalVariable(name: "p",

declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !1000}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 348209)"}
!7 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !1, file: !1, line: 4, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "p", arg: 1, scope: !7, file: !1, line: 4, type: !10)
!14 = !DILocation(line: 4, column: 15, scope: !7)
!20 = distinct !DILexicalBlock(scope: !7, file: !1, line: 8, column: 7)
!21 = distinct !DIAssignID()
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
