; RUN: llc -mtriple=aarch64-unknown-linux-gnu -O2 -stop-after=virtregrewriter %s -o - | FileCheck %s

; LiveDebugVariables records SlotIndexes during register allocation and uses
; them afterwards in emitDebugValues, but nothing tells it when an instruction
; is erased in between, so a recorded index can be left naming an instruction
; that no longer exists.
;
; Here HoistSpillHelper::hoistAllSpills removes a redundant spill, and the
; interval bounds in UserValue::locInts are left naming it. The location still
; comes out in the right place because findInsertLocation scans backwards from
; the recorded index for a surviving instruction to insert after, so this test
; pins down that currently correct output.
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

!llvm.dbg.cu = !{!0}
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
