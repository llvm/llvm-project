; RUN: opt -passes=redundant-dbg-inst-elim -S %s -o - \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"

;; Hand-written. Test how RemoveRedundantDbgInstrs interacts with dbg.assign
;; intrinsics. FileCehck directives are inline.

define dso_local void @_Z1fv() !dbg !7 {
entry:
  %test = alloca i32, align 4, !DIAssignID !20
; CHECK: alloca
;; Forward scan: This dbg.assign for Local2 contains an undef value component
;; in the entry block and is the first debug intrinsic for the variable, but is
;; linked to an instruction so should not be deleted.
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i1 undef, metadata ![[Local2:[0-9]+]]
  call void @llvm.dbg.assign(metadata i1 undef, metadata !19, metadata !DIExpression(), metadata !20, metadata ptr %test, metadata !DIExpression()), !dbg !14

;; Forward scan: dbg.assign for Local unlinked with undef value component, in
;; the enrty bock and seen before any non-undefs; delete it.
; CHECK-NEXT: @step()
  call void @llvm.dbg.assign(metadata i32 undef, metadata !11, metadata !DIExpression(), metadata !15, metadata ptr undef, metadata !DIExpression()), !dbg !14
  call void @step()

;; Forward scan: Repeat the previous test to check it works more than once.
; CHECK-NEXT: @step()
  call void @llvm.dbg.assign(metadata i32 undef, metadata !11, metadata !DIExpression(), metadata !15, metadata ptr undef, metadata !DIExpression()), !dbg !14
  call void @step()

;; Backward scan: Check that a dbg.value made redundant by a dbg.assign is
;; removed.
;; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 1, metadata ![[Local:[0-9]+]]
;; CHECK-NEXT: @step()
  call void @llvm.dbg.value(metadata i32 0, metadata !11, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.assign(metadata i32 1, metadata !11, metadata !DIExpression(), metadata !15, metadata ptr undef, metadata !DIExpression()), !dbg !14
  call void @step()

;; Backward scan: Check that a dbg.assign made redundant by a dbg.value is
;; removed.
;; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 3, metadata ![[Local:[0-9]+]]
;; CHECK-NEXT: @step()
  call void @llvm.dbg.assign(metadata i32 2, metadata !11, metadata !DIExpression(), metadata !15, metadata ptr undef, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.value(metadata i32 3, metadata !11, metadata !DIExpression()), !dbg !14
  call void @step()

;; Forward scan: This unlinked dbg.assign(3, ...) is shadowed by the
;; dbg.value(3,...) above. Check it is removed.
;; CHECK-NEXT: @step()
  call void @llvm.dbg.assign(metadata i32 3, metadata !11, metadata !DIExpression(), metadata !15, metadata ptr undef, metadata !DIExpression()), !dbg !14
  call void @step()

;; Forward scan: Same as above except this dbg.assign is shadowed by
;; another dbg.assign rather than a dbg.value. Check it is removed.
;; CHECK-NEXT: @step()
  call void @llvm.dbg.assign(metadata i32 3, metadata !11, metadata !DIExpression(), metadata !15, metadata ptr undef, metadata !DIExpression()), !dbg !14
  call void @step()

;; Forward scan: We've seen non-undef dbg intrinsics for Local in the entry
;; block so we shouldn't delete this undef.
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 undef, metadata ![[Local]]
  call void @llvm.dbg.assign(metadata i32 undef, metadata !11, metadata !DIExpression(), metadata !15, metadata ptr undef, metadata !DIExpression()), !dbg !14
  br label %next

next:
;; Forward scan: Do not delete undef dbg.assigns from non-entry blocks.
; CHECK: call void @llvm.dbg.assign(metadata i32 undef, metadata ![[Local2]]
; CHECK-NEXT: @step()
  call void @llvm.dbg.assign(metadata i32 undef, metadata !19, metadata !DIExpression(), metadata !21, metadata ptr %test, metadata !DIExpression()), !dbg !14
  call void @step()

;; Forward scan: The next dbg.assign would be made redundant by this dbg.value
;; if it were not for the fact that it is linked to an instruction. Ensure it
;; isn't removed.
;; Backward scan: It (the next dbg.assign) is also followed by another for the
;; same variable - check it isn't remove (because it's linked).
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 0, metadata ![[Local2]]
; CHECK-NEXT: store
; CHECK-NEXT: store
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 0, metadata ![[Local2]]
; CHECK-NEXT: call void @llvm.dbg.assign(metadata i32 1, metadata ![[Local2]]
  call void @llvm.dbg.value(metadata i32 0, metadata !19, metadata !DIExpression()), !dbg !14
  store i32 0, ptr %test, !DIAssignID !17
  store i32 1, ptr %test, !DIAssignID !16
  call void @llvm.dbg.assign(metadata i32 0, metadata !19, metadata !DIExpression(), metadata !17, metadata ptr %test, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.assign(metadata i32 1, metadata !19, metadata !DIExpression(), metadata !16, metadata ptr %test, metadata !DIExpression()), !dbg !14
  ret void, !dbg !18
}

; CHECK-DAG: ![[Local2]] = !DILocalVariable(name: "Local2",
; CHECK-DAG: ![[Local]] = !DILocalVariable(name: "Local",

declare dso_local void @step()
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !1000}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{!"clang version 14.0.0"}
!7 = distinct !DISubprogram(name: "f", linkageName: "_Z1fv", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11, !19}
!11 = !DILocalVariable(name: "Local", scope: !7, file: !1, line: 2, type: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = distinct !DIAssignID()
!14 = !DILocation(line: 0, scope: !7)
!15 = distinct !DIAssignID()
!16 = distinct !DIAssignID()
!17 = distinct !DIAssignID()
!18 = !DILocation(line: 6, column: 1, scope: !7)
!19 = !DILocalVariable(name: "Local2", scope: !7, file: !1, line: 2, type: !12)
!20 = distinct !DIAssignID()
!21 = distinct !DIAssignID()
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
