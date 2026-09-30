; RUN: opt -passes=hotcoldsplit -hotcoldsplit-threshold=-1 -S < %s | FileCheck %s

; The outlined function describes each variable as it was on entry to the
; region, taken from the last record for it before the region.

declare void @sink(i32) cold
declare void @sinkp(ptr) cold

; v == a on entry. The record for v == x comes after the region, so it is not
; copied, and a is not passed in, so v has no location.
; CHECK-LABEL: define internal void @after_region.cold.1(i32 %x)
; CHECK-NOT: #dbg_value
; CHECK: ret void
define void @after_region(i32 %a, i1 %c) !dbg !10 {
entry:
  %x = add i32 %a, 1, !dbg !11
    #dbg_value(i32 %a, !12, !DIExpression(), !11)
  br i1 %c, label %cold, label %exit, !dbg !11
cold:
  %m = mul i32 %x, 3, !dbg !11
  call void @sink(i32 %m), !dbg !11
  br label %exit, !dbg !11
exit:
    #dbg_value(i32 %x, !12, !DIExpression(), !11)
  ret void, !dbg !11
}

; Only the later of two records before the region is copied.
; CHECK-LABEL: define internal void @superseded.cold.1(i32 %x, i32 %y)
; CHECK-NEXT: newFuncRoot:
; CHECK-NEXT: #dbg_value(i32 %y,
; CHECK-NEXT: br label
define void @superseded(i32 %a, i1 %c) !dbg !20 {
entry:
  %x = add i32 %a, 1, !dbg !21
  %y = add i32 %a, 2, !dbg !21
    #dbg_value(i32 %x, !22, !DIExpression(), !21)
    #dbg_value(i32 %y, !22, !DIExpression(), !21)
  br i1 %c, label %cold, label %exit, !dbg !21
cold:
  %s = add i32 %x, %y, !dbg !21
  call void @sink(i32 %s), !dbg !21
  br label %exit, !dbg !21
exit:
  ret void, !dbg !21
}

; The two ways into the region disagree about v, so v has no location.
; CHECK-LABEL: define internal void @merge.cold.1(i32 %x, i32 %y)
; CHECK-NOT: #dbg_value
; CHECK: ret void
define void @merge(i32 %a, i1 %c, i1 %d) !dbg !30 {
entry:
  %x = add i32 %a, 1, !dbg !31
  %y = add i32 %a, 2, !dbg !31
  br i1 %d, label %l, label %r, !dbg !31
l:
    #dbg_value(i32 %x, !32, !DIExpression(), !31)
  br i1 %c, label %cold, label %exit, !dbg !31
r:
    #dbg_value(i32 %y, !32, !DIExpression(), !31)
  br i1 %c, label %cold, label %exit, !dbg !31
cold:
  %s = add i32 %x, %y, !dbg !31
  call void @sink(i32 %s), !dbg !31
  br label %exit, !dbg !31
exit:
  ret void, !dbg !31
}

; A constant is valid in any function.
; CHECK-LABEL: define internal void @constant.cold.1(i32 %x)
; CHECK: #dbg_value(i32 42,
define void @constant(i32 %a, i1 %c) !dbg !40 {
entry:
  %x = add i32 %a, 1, !dbg !41
    #dbg_value(i32 42, !42, !DIExpression(), !41)
  br i1 %c, label %cold, label %exit, !dbg !41
cold:
  %m = mul i32 %x, 3, !dbg !41
  call void @sink(i32 %m), !dbg !41
  br label %exit, !dbg !41
exit:
  ret void, !dbg !41
}

; A declare holds for the whole function, so a merge point does not matter.
; CHECK-LABEL: define internal void @declare_behind_merge.cold.1(ptr %p)
; CHECK: #dbg_declare(ptr %p, ![[#]], !DIExpression(DW_OP_plus_uconst, 4),
define void @declare_behind_merge(i1 %c, i1 %d) !dbg !50 {
entry:
  %p = alloca i32, align 4, !dbg !51
    #dbg_declare(ptr %p, !52, !DIExpression(DW_OP_plus_uconst, 4), !51)
  br i1 %d, label %l, label %r, !dbg !51
l:
  br i1 %c, label %cold, label %exit, !dbg !51
r:
  br i1 %c, label %cold, label %exit, !dbg !51
cold:
  call void @sinkp(ptr %p), !dbg !51
  br label %exit, !dbg !51
exit:
  ret void, !dbg !51
}

; Both fragments are copied. The whole-variable record before them is not, even
; though a is passed in, because the fragments cover it.
; CHECK-LABEL: define internal void @fragments.cold.1(i32 %x, i32 %y, i64 %a)
; CHECK-NEXT: newFuncRoot:
; CHECK-NEXT: #dbg_value(i32 %x, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 0, 32),
; CHECK-NEXT: #dbg_value(i32 %y, ![[#]], !DIExpression(DW_OP_LLVM_fragment, 32, 32),
; CHECK-NEXT: br label
define void @fragments(i64 %a, i1 %c) !dbg !60 {
entry:
  %t = trunc i64 %a to i32, !dbg !61
  %x = add i32 %t, 1, !dbg !61
  %y = add i32 %t, 2, !dbg !61
    #dbg_value(i64 %a, !62, !DIExpression(), !61)
    #dbg_value(i32 %x, !62, !DIExpression(DW_OP_LLVM_fragment, 0, 32), !61)
    #dbg_value(i32 %y, !62, !DIExpression(DW_OP_LLVM_fragment, 32, 32), !61)
  br i1 %c, label %cold, label %exit, !dbg !61
cold:
  %s = add i32 %x, %y, !dbg !61
  %a32 = trunc i64 %a to i32, !dbg !61
  %r = add i32 %s, %a32, !dbg !61
  call void @sink(i32 %r), !dbg !61
  br label %exit, !dbg !61
exit:
  ret void, !dbg !61
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !2)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!10 = distinct !DISubprogram(name: "after_region", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!11 = !DILocation(line: 1, column: 1, scope: !10)
!12 = !DILocalVariable(name: "v", scope: !10, file: !1, line: 1, type: !5)
!20 = distinct !DISubprogram(name: "superseded", scope: !1, file: !1, line: 2, type: !4, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!21 = !DILocation(line: 2, column: 1, scope: !20)
!22 = !DILocalVariable(name: "v", scope: !20, file: !1, line: 2, type: !5)
!30 = distinct !DISubprogram(name: "merge", scope: !1, file: !1, line: 3, type: !4, scopeLine: 3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!31 = !DILocation(line: 3, column: 1, scope: !30)
!32 = !DILocalVariable(name: "v", scope: !30, file: !1, line: 3, type: !5)
!40 = distinct !DISubprogram(name: "constant", scope: !1, file: !1, line: 4, type: !4, scopeLine: 4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!41 = !DILocation(line: 4, column: 1, scope: !40)
!42 = !DILocalVariable(name: "v", scope: !40, file: !1, line: 4, type: !5)
!50 = distinct !DISubprogram(name: "declare_behind_merge", scope: !1, file: !1, line: 5, type: !4, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!51 = !DILocation(line: 5, column: 1, scope: !50)
!52 = !DILocalVariable(name: "p", scope: !50, file: !1, line: 5, type: !5)
!60 = distinct !DISubprogram(name: "fragments", scope: !1, file: !1, line: 6, type: !4, scopeLine: 6, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!61 = !DILocation(line: 6, column: 1, scope: !60)
!62 = !DILocalVariable(name: "w", scope: !60, file: !1, line: 6, type: !6)
