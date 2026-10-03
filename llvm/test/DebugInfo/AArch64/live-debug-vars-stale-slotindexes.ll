; RUN: llc -mtriple=aarch64-unknown-linux-gnu -O2 -stop-after=virtregrewriter %s -o - | FileCheck %s

; LiveDebugVariables records SlotIndexes during register allocation and uses
; them afterwards to re-insert debug instructions, but nothing tells it when an
; instruction is erased in between, so a recorded index can be left naming an
; instruction that no longer exists. Each function below reaches one such case.
; The debug instructions still come out in the right place, so these tests pin
; down that currently correct output.
;
; See also DebugInfo/X86/live-debug-vars-stale-slotindexes.ll.

; HoistSpillHelper::hoistAllSpills removes a redundant spill, and the interval
; bounds in UserValue::locInts are left naming it. findInsertLocation scans
; backwards from the recorded index for a surviving instruction to insert
; after.
;
; Reduced from MultiSource/Benchmarks/7zip 7zUpdate.cpp built at -O2 -g.

; CHECK-LABEL: name: update
; CHECK: DBG_VALUE $x1, $noreg, ![[VAR:[0-9]+]], !DIExpression(DW_OP_plus_uconst, 72, DW_OP_stack_value)
; CHECK: DBG_VALUE %stack.[[#]], 0, ![[VAR]], !DIExpression(DW_OP_plus_uconst, 72, DW_OP_stack_value)

define i32 @update(ptr %db, ptr %mode, i1 %c1, ptr %p0, i1 %c2, ptr %p1, ptr %p2, ptr %p3, i1 %c3) !dbg !4 {
entry:
  %pw = getelementptr i8, ptr %mode, i64 72
  br label %loop.outer

loop.outer:                                       ; preds = %latch, %entry
    #dbg_value(ptr %pw, !9, !DIExpression(), !17)
  call void @llvm.memset.p0.i64(ptr %mode, i8 0, i64 1, i1 false)
  br i1 %c3, label %for.cond, label %common.ret

common.ret:                                       ; preds = %for.body, %loop.outer
  ret i32 0

for.cond:                                         ; preds = %for.end, %loop.outer
  br i1 %c1, label %for.body, label %latch

for.body:                                         ; preds = %for.cond
  %v = load volatile ptr, ptr null, align 8
  br i1 %c2, label %for.end, label %common.ret

for.end:                                          ; preds = %for.body
  %call = call ptr @allocate(i64 0)
  store i64 4, ptr %p1, align 8
  store ptr %p2, ptr %p0, align 8
  store i64 1, ptr %p3, align 8
  store volatile i32 0, ptr null, align 4
  store i32 0, ptr %db, align 4
  br label %for.cond

latch:                                            ; preds = %for.cond
  store volatile i32 0, ptr null, align 4
  br label %loop.outer
}

declare ptr @allocate(i64)

; UserValue::trimmedDefs records interval start indexes that computeIntervals
; trimmed to a lexical scope, and emitDebugValues matches them against the
; starts in locInts to decide whether to step a DBG_VALUE back one index. Here
; both a trimmed start and the locInts start it belongs to are left naming an
; erased instruction.
;
; Reduced from MultiSource/Benchmarks/DOE-ProxyApps-C++/PENNANT built at -O2 -g.

; CHECK-LABEL: name: map_index
; CHECK: DBG_VALUE $x0, $noreg, ![[VAR:[0-9]+]], !DIExpression()
; CHECK: DBG_VALUE $x{{[0-9]+}}, $noreg, ![[VAR]], !DIExpression()

define ptr @map_index(ptr %this, ptr %k, ptr %p0, ptr %x.addr.1, ptr %p1, i64 %n, i1 %cmp.i) !dbg !24 {
entry:
    #dbg_value(ptr %this, !30, !DIExpression(), !37)
  %add.ptr = getelementptr i8, ptr %this, i64 8, !dbg !47
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %x.addr.011 = phi ptr [ null, %entry ], [ %x.addr.1, %while.body ]
  %call = call i32 @memcmp(ptr %p1)
  %sub = sub i64 0, %n
  %spec.select = tail call i64 @llvm.smax.i64(i64 %sub, i64 1)
  %retval.07 = tail call i64 @llvm.smin.i64(i64 %spec.select, i64 2147483647)
  br i1 %cmp.i, label %lower_bound.exit, label %while.body

lower_bound.exit:                                 ; preds = %while.body
  %retval.0.i12 = trunc i64 %retval.07 to i32
  %cmp.i3 = icmp slt i32 %retval.0.i12, 0
  %y.addr.1 = select i1 %cmp.i3, ptr null, ptr %x.addr.011
  %cmp = icmp eq ptr %y.addr.1, %add.ptr
  br i1 %cmp, label %if.then, label %common.ret

common.ret:                                       ; preds = %if.then, %lower_bound.exit
  ret ptr null

if.then:                                          ; preds = %lower_bound.exit
  store ptr %k, ptr %p0, align 8
  %call12 = load volatile ptr, ptr %this, align 8
  br label %common.ret
}

declare i32 @memcmp(ptr)

!llvm.dbg.cu = !{!0, !20}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "7zUpdate.cpp", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "Update", scope: !1, file: !1, line: 714, type: !8, scopeLine: 728, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!8 = distinct !DISubroutineType(types: !2)
!9 = !DILocalVariable(name: "this", arg: 1, scope: !10, type: !16, flags: DIFlagArtificial | DIFlagObjectPointer)
!10 = distinct !DISubprogram(name: "CStringBase", scope: !1, file: !1, line: 182, type: !8, scopeLine: 182, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!17 = distinct !DILocation(line: 0, scope: !10, inlinedAt: !18)
!18 = distinct !DILocation(line: 41, column: 3, scope: !19)
!19 = distinct !DISubprogram(name: "CCompressionMethodMode", scope: !1, file: !1, line: 41, type: !8, scopeLine: 45, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!20 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !21, producer: "clang", isOptimized: true, emissionKind: FullDebug)
!21 = !DIFile(filename: "InputFile.cc", directory: "/")
!22 = !{}
!24 = distinct !DISubprogram(name: "operator[]", scope: !26, file: !25, line: 504, type: !28, scopeLine: 505, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !20)
!25 = !DIFile(filename: "stl_map.h", directory: "/")
!26 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "map", scope: !27, file: !25, line: 102, size: 384, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !22)
!27 = !DINamespace(name: "std", scope: null)
!28 = distinct !DISubroutineType(types: !22)
!30 = !DILocalVariable(name: "this", arg: 1, scope: !31, type: !36, flags: DIFlagArtificial | DIFlagObjectPointer)
!31 = distinct !DISubprogram(name: "_M_end", scope: !33, file: !32, line: 747, type: !34, scopeLine: 748, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !20)
!32 = !DIFile(filename: "stl_tree.h", directory: "/")
!33 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "_Rb_tree", scope: !27, file: !32, line: 423, size: 384, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !22)
!34 = distinct !DISubroutineType(types: !22)
!36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !33, size: 64)
!37 = distinct !DILocation(line: 0, scope: !31, inlinedAt: !38)
!38 = distinct !DILocation(line: 1267, column: 43, scope: !39, inlinedAt: !42)
!39 = distinct !DISubprogram(name: "lower_bound", scope: !33, file: !32, line: 1266, type: !40, scopeLine: 1267, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !20)
!40 = distinct !DISubroutineType(types: !22)
!42 = distinct !DILocation(line: 1308, column: 21, scope: !43, inlinedAt: !46)
!43 = distinct !DISubprogram(name: "lower_bound", scope: !26, file: !25, line: 1307, type: !44, scopeLine: 1308, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !20)
!44 = distinct !DISubroutineType(types: !22)
!46 = distinct !DILocation(line: 509, column: 17, scope: !24)
!47 = distinct !DILocation(line: 748, column: 17, scope: !31, inlinedAt: !38)
