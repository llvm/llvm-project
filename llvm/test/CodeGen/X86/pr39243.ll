; RUN: opt < %s -S -passes=simplifycfg | FileCheck %s

; Note: This patch fixes the regression introduced by pr38762.
;
; When SimplifyCFG changes the PHI node into a select instruction, the debug
; information becomes ambiguous. It causes the debugger to display unreached
; lines and invalid variable values.
;
; When the function 'hoistAllInstructionsInto' hoist a basic block:
; - Remove their dbg.values.
; - Set their debug locations to the values from the insertion point.
;
; But, if one of the instructions being hoisted is a debug intrinsic from an
; inlined function, assigning it the debug location from the insertion point
; will create a mismatch between the intrinsic's subprogram and the location's
; subprogram, causing the assertion "Expected inlined-at fields to agree" in
; SelectionDAG".

; IR generated with:
; clang -S -g -gno-column-info -O2 -emit-llvm pr39243.cpp -o pr39243.ll -mllvm -opt-bisect-limit=103

; // pr39243.cpp
; union onion {
;   double dd;
;   int ii[2];
; };
;
; int alpha;
; int bravo();
;
; int charlie() {
;   int delta = 0;
;   return bravo();
; }
;
; int echo(onion foxtrot) {
;   alpha = foxtrot.ii[0];
;   if (alpha) {
;     int golf = bravo();
;     return -golf;
;   }
;   alpha = foxtrot.ii[1];
;   return -charlie();
; }

; Change the debug locations associated with the PHI nodes being promoted, to
; the debug locations from the insertion point in the dominant block.

; CHECK-LABEL: entry
; CHECK:  %foxtrot.sroa.0.0.extract.trunc = trunc i64 %foxtrot.coerce to i32
; CHECK:  %tobool = icmp eq i32 %foxtrot.sroa.0.0.extract.trunc, 0
; CHECK:  %foxtrot.sroa.2.0.extract.shift = lshr i64 %foxtrot.coerce, 32
; CHECK-NOT:  call void @llvm.dbg.value(metadata i32 %foxtrot.sroa.2.0.extract.trunc, metadata !30, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32)), !dbg !34
; CHECK:  %foxtrot.sroa.2.0.extract.trunc = trunc i64 %foxtrot.sroa.2.0.extract.shift to i32
; CHECK:  store i32 %foxtrot.sroa.2.0.extract.trunc, ptr @alpha, align 4, !dbg !25
; CHECK-NOT:  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !43

; ModuleID = 'pr39243.cpp'
source_filename = "pr39243.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@alpha = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0

define dso_local i32 @_Z7charliev() local_unnamed_addr #0 {
entry:
  %call = tail call i32 @_Z5bravov()
  ret i32 %call
}

declare dso_local i32 @_Z5bravov() local_unnamed_addr #1

define dso_local i32 @_Z4echo5onion(i64 %foxtrot.coerce) local_unnamed_addr #0 !dbg !18 {
entry:
  %foxtrot.sroa.0.0.extract.trunc = trunc i64 %foxtrot.coerce to i32
  store i32 %foxtrot.sroa.0.0.extract.trunc, ptr @alpha, align 4
  %tobool = icmp eq i32 %foxtrot.sroa.0.0.extract.trunc, 0
  br i1 %tobool, label %if.end, label %return

if.end:                                           ; preds = %entry
  %foxtrot.sroa.2.0.extract.shift = lshr i64 %foxtrot.coerce, 32
  %foxtrot.sroa.2.0.extract.trunc = trunc i64 %foxtrot.sroa.2.0.extract.shift to i32
  call void @llvm.dbg.value(metadata i32 %foxtrot.sroa.2.0.extract.trunc, metadata !30, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32)), !dbg !34
  store i32 %foxtrot.sroa.2.0.extract.trunc, ptr @alpha, align 4, !dbg !42
  call void @llvm.dbg.value(metadata i32 0, metadata !15, metadata !DIExpression()), !dbg !43
  br label %return

return:                                           ; preds = %entry, %if.end
  %call.i = tail call i32 @_Z5bravov()
  %retval.0 = sub nsw i32 0, %call.i
  ret i32 %retval.0
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #2

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "alpha", scope: !2, file: !3, line: 6, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 8.0.0 (trunk 344502)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "pr39243.cpp", directory: ".")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 8.0.0 (trunk 344502)"}
!11 = distinct !DISubprogram(name: "charlie", linkageName: "_Z7charliev", scope: !3, file: !3, line: 9, type: !12, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !14)
!12 = !DISubroutineType(types: !13)
!13 = !{!6}
!14 = !{!15}
!15 = !DILocalVariable(name: "delta", scope: !11, file: !3, line: 10, type: !6)
!18 = distinct !DISubprogram(name: "echo", linkageName: "_Z4echo5onion", scope: !3, file: !3, line: 14, type: !19, isLocal: false, isDefinition: true, scopeLine: 14, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !29)
!19 = !DISubroutineType(types: !20)
!20 = !{!6, !21}
!21 = distinct !DICompositeType(tag: DW_TAG_union_type, name: "onion", file: !3, line: 1, size: 64, flags: DIFlagTypePassByValue, elements: !22, identifier: "_ZTS5onion")
!22 = !{!23, !25}
!23 = !DIDerivedType(tag: DW_TAG_member, name: "dd", scope: !21, file: !3, line: 2, baseType: !24, size: 64)
!24 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "ii", scope: !21, file: !3, line: 3, baseType: !26, size: 64)
!26 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 64, elements: !27)
!27 = !{!28}
!28 = !DISubrange(count: 2)
!29 = !{!30}
!30 = !DILocalVariable(name: "foxtrot", arg: 1, scope: !18, file: !3, line: 14, type: !21)
!34 = !DILocation(line: 14, scope: !18)
!42 = !DILocation(line: 20, scope: !18)
!43 = !DILocation(line: 10, scope: !11, inlinedAt: !44)
!44 = distinct !DILocation(line: 21, scope: !18)
