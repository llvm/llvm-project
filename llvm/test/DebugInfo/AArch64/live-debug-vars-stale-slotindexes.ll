; RUN: llc -mtriple=aarch64-unknown-linux-gnu -stop-after=virt-reg-rewriter \
; RUN:   %s -o - | FileCheck %s --implicit-check-not=DBG_VALUE

; LiveDebugVariables records SlotIndexes during register allocation and uses
; them afterwards to re-insert debug instructions. Passes in between can erase
; the instructions those indexes name, and canonicalizeIndexes() repairs such
; stale indexes to the closest preceding live instruction. Each function below
; reaches it by a different route. Check that each debug instruction still comes
; out once, in the right place.
;
; See also DebugInfo/X86/live-debug-vars-stale-slotindexes.ll.

; Splitting the register holding %pw around its spill leaves two intervals in
; UserValue::locInts, one in $x1 and one on the stack, with the boundary between
; them at the spill. HoistSpillHelper::hoistAllSpills hoists the spill into the
; entry block, erasing the original. That boundary resolves to the start of
; loop.outer, which empties the $x1 interval, so only the stack location is
; emitted.
;
; Reduced from MultiSource/Benchmarks/7zip 7zUpdate.cpp built at -O2 -g.

; CHECK-LABEL: name: update
; CHECK:       bb.0.entry:
; CHECK:       STRXui renamable $x1, %stack.0, 0
; CHECK:       bb.1.loop.outer:
; CHECK:       liveins:
; CHECK-NEXT:  {{^ *$}}
; CHECK-NEXT:  DBG_VALUE %stack.0, 0, !{{[0-9]+}}, !DIExpression(DW_OP_plus_uconst, 72, DW_OP_stack_value)
; CHECK-NEXT:  STRBBui $wzr, killed renamable $x1, 0

define i32 @update(ptr %db, ptr %mode, i1 %c1, ptr %p0, i1 %c2, ptr %p1, ptr %p2, ptr %p3, i1 %c3) !dbg !4 {
entry:
  %pw = getelementptr i8, ptr %mode, i64 72
  br label %loop.outer

loop.outer:                                       ; preds = %latch, %entry
    #dbg_value(ptr %pw, !6, !DIExpression(), !9)
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
; both a trimmed start and the locInts start it belongs to name an erased
; instruction, so they have to be repaired consistently to keep matching.
;
; Reduced from MultiSource/Benchmarks/DOE-ProxyApps-C++/PENNANT built at -O2 -g.

; CHECK-LABEL: name: map_index
; CHECK:       bb.0.entry:
; CHECK:       liveins:
; CHECK-NEXT:  {{^ *$}}
; CHECK-NEXT:  DBG_VALUE $x0, $noreg, ![[VAR:[0-9]+]], !DIExpression()
; CHECK:       renamable $x[[THIS:[0-9]+]] = COPY $x0
; CHECK-NEXT:  renamable $x8 = COPY $xzr
; CHECK-NEXT:  DBG_VALUE $x[[THIS]], $noreg, ![[VAR]], !DIExpression()
; CHECK-NEXT:  SUBXrr $xzr

define ptr @map_index(ptr %this, ptr %k, ptr %p0, ptr %x.addr.1, ptr %p1, i64 %n, i1 %cmp.i) !dbg !14 {
entry:
    #dbg_value(ptr %this, !18, !DIExpression(), !23)
  %add.ptr = getelementptr i8, ptr %this, i64 8, !dbg !29
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %x.addr.011 = phi ptr [ null, %entry ], [ %x.addr.1, %while.body ]
  %call = call i32 @cmp(ptr %p1)
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

declare i32 @cmp(ptr)

!llvm.dbg.cu = !{!0, !12}
!llvm.module.flags = !{!3}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "7zUpdate.cpp", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "Update", scope: !1, file: !1, line: 714, type: !5, scopeLine: 728, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !2)
!6 = !DILocalVariable(name: "this", arg: 1, scope: !7, type: !8)
!7 = distinct !DISubprogram(name: "CStringBase", scope: !1, file: !1, line: 182, type: !5, scopeLine: 182, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!9 = !DILocation(line: 0, scope: !7, inlinedAt: !10)
!10 = !DILocation(line: 41, column: 3, scope: !11)
!11 = distinct !DISubprogram(name: "CCompressionMethodMode", scope: !1, file: !1, line: 41, type: !5, scopeLine: 45, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !13, producer: "clang", isOptimized: true, emissionKind: FullDebug)
!13 = !DIFile(filename: "InputFile.cc", directory: "/")
!14 = distinct !DISubprogram(name: "operator[]", scope: !16, file: !15, line: 504, type: !5, scopeLine: 505, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !12)
!15 = !DIFile(filename: "stl_map.h", directory: "/")
!16 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "map", scope: !17, file: !15, line: 102, size: 384, elements: !2)
!17 = !DINamespace(name: "std", scope: null)
!18 = !DILocalVariable(name: "this", arg: 1, scope: !19, type: !22)
!19 = distinct !DISubprogram(name: "_M_end", scope: !21, file: !20, line: 747, type: !5, scopeLine: 748, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !12)
!20 = !DIFile(filename: "stl_tree.h", directory: "/")
!21 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "_Rb_tree", scope: !17, file: !20, line: 423, size: 384, elements: !2)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64)
!23 = !DILocation(line: 0, scope: !19, inlinedAt: !24)
!24 = !DILocation(line: 1267, column: 43, scope: !25, inlinedAt: !26)
!25 = distinct !DISubprogram(name: "lower_bound", scope: !21, file: !20, line: 1266, type: !5, scopeLine: 1267, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !12)
!26 = !DILocation(line: 1308, column: 21, scope: !27, inlinedAt: !28)
!27 = distinct !DISubprogram(name: "lower_bound", scope: !16, file: !15, line: 1307, type: !5, scopeLine: 1308, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !12)
!28 = !DILocation(line: 509, column: 17, scope: !14)
!29 = !DILocation(line: 748, column: 17, scope: !19, inlinedAt: !24)
