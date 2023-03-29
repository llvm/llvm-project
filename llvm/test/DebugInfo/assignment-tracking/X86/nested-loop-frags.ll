; RUN: llc %s -stop-after=finalize-isel -o - \
; RUN: | FileCheck %s --implicit-check-not=DBG

;; Test a variety of block inputs and lattice configurations for the assignment
;; tracking analysis (debug-ata). This is similar to nested-loop.ll and
;; nested-loop-sroa.ll except that the allocas are 64 bits and the stores are a
;; mix of 32 and 64 bits. Unlike nested-loop-sroa.ll this one is not a clone
;; of nested-loop.ll.

;; The CFG looks like this:
;;     entry
;;     |
;;     v
;;     do.body <-----+
;;     |             |
;;     V             |
;;     do.body1 <--+ |
;;    / \          | |
;;   /   \         | |
;;  /     \        | |
;; v       v       | |
;; if.then if.else | |
;;  \      /       | |
;;   \    /        | |
;;    \  /         | |
;;     do.cond ----+ |
;;     |             |
;;     v             |
;;     do.cond4 -----+
;;     |
;;     v
;;     do.end6

;; This version doesn't contain tables of assignments as it is difficult to neatly represent
;; the additional dimension of fragments (stores to part of the variable/alloca).

;; Variable 'a' (!21)
;; Check that both the full assignment (!70) assignment to the lower bits (and !63) are
;; propagated to do.end6.
;;
;; Variable 'b' (!22)
;; Check mem=dbg assignment to the lower 32 bits in if.then causes a mem=phi (tested by
;; looking for value-based DBG_VALUE in do.end6). Meanwhile, check the assignment (!71)
;; is propagated to do.end6 for the upper bits (checked by looking for a memory location).
;;
;; Variable 'c' (!67)
;; Check initial dbg and mem assignment values are propagated through all blocks, with
;; dbg defs with the inital assignment ID put in do.cond and do.end6 (variable is always
;; in memory). The initial mem and dbg defs are to the whole variable, and the subsequent
;; dbg defs come in pairs, split for the high and low bits of the variable.
;;
;; Variable 'd' (!72)
;; Same as above, except the dbg def in do.cond has been split into two assignments
;; that both use the same ID as the inital one - one fragment is assigned in each of
;; the if-branches.
;;
;; Variable 'e' (!75)
;; The join in do.body covers assignments to different fragments. Out of entry
;; we have [0-31: mem=!77 dbg=!78 loc=val] [32-63: mem=!77 dbg=!76 loc=val].
;; There's a tagged store to the lower bits in if.then and an untagged store to
;; the upper bits in if.else. The important check here is that in do.body the
;; memory location isn't used at the dbg def !77 as the live-in loc=val and the
;; incoming mem assignments are not all !77.

; CHECK-DAG: ![[a:[0-9]+]] = !DILocalVariable(name: "a",
; CHECK-DAG: ![[b:[0-9]+]] = !DILocalVariable(name: "b",
; CHECK-DAG: ![[c:[0-9]+]] = !DILocalVariable(name: "c",
; CHECK-DAG: ![[d:[0-9]+]] = !DILocalVariable(name: "d",
; CHECK-DAG: ![[e:[0-9]+]] = !DILocalVariable(name: "e",

;; Variables 'c' (!67) and 'd' (!72) are always stack-homed.
; CHECK:      - { id: 2, name: c.addr, type: default, offset: 0, size: 8, alignment: 8,
; CHECK-NEXT:     stack-id: default, callee-saved-register: '', callee-saved-restored: true,
; CHECK-NEXT:     debug-info-variable: '![[c]]', debug-info-expression: '!DIExpression()',
; CHECK:      - { id: 3, name: d.addr, type: default, offset: 0, size: 8, alignment: 8,
; CHECK-NEXT:     stack-id: default, callee-saved-register: '', callee-saved-restored: true,
; CHECK-NEXT:     debug-info-variable: '![[d]]', debug-info-expression: '!DIExpression()',

source_filename = "test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g_a = dso_local local_unnamed_addr global i32 0, align 4, !dbg !0
@g_b = dso_local local_unnamed_addr global i32 0, align 4, !dbg !5
@g_c = dso_local local_unnamed_addr global i32 0, align 4, !dbg !8

define dso_local noundef i32 @_Z3funii(i32 noundef %a, i32 noundef %b) local_unnamed_addr #0 !dbg !17 {
entry:
  %a.addr = alloca i64, align 8, !DIAssignID !58 ; VAR:a
  call void @llvm.dbg.assign(metadata i1 undef, metadata !21, metadata !DIExpression(), metadata !58, metadata ptr %a.addr, metadata !DIExpression()), !dbg !27 ; VAR:a
  %b.addr = alloca i64, align 8, !DIAssignID !64 ; VAR:b
  call void @llvm.dbg.assign(metadata i1 undef, metadata !22, metadata !DIExpression(), metadata !64, metadata ptr %b.addr, metadata !DIExpression()), !dbg !27 ; VAR:b
  %c.addr = alloca i64, align 8, !DIAssignID !68 ; VAR:c
  call void @llvm.dbg.assign(metadata i1 undef, metadata !67, metadata !DIExpression(), metadata !68, metadata ptr %c.addr, metadata !DIExpression()), !dbg !27 ; VAR:c
  %d.addr = alloca i64, align 8, !DIAssignID !73 ; VAR:d
  call void @llvm.dbg.assign(metadata i1 undef, metadata !72, metadata !DIExpression(), metadata !73, metadata ptr %d.addr, metadata !DIExpression()), !dbg !27 ; VAR:d
  %e.addr = alloca i64, align 8, !DIAssignID !76 ; VAR:e
  call void @llvm.dbg.assign(metadata i1 undef, metadata !75, metadata !DIExpression(), metadata !76, metadata ptr %e.addr, metadata !DIExpression()), !dbg !27 ; VAR:e
  ;%f.addr = alloca i64, align 8, !DIAssignID !80 ; VAR:f
  ;call void @llvm.dbg.assign(metadata i1 undef, metadata !79, metadata !DIExpression(), metadata !80, metadata ptr %f.addr, metadata !DIExpression()), !dbg !27 ; VAR:f
  store i64 1, ptr %a.addr, !DIAssignID !70 ; VAR:a
  call void @llvm.dbg.assign(metadata i64 1, metadata !21, metadata !DIExpression(), metadata !70, metadata ptr %a.addr, metadata !DIExpression()), !dbg !27 ; VAR:a
  store i64 2, ptr %b.addr, !DIAssignID !71 ; VAR:b
  call void @llvm.dbg.assign(metadata i32 2, metadata !22, metadata !DIExpression(), metadata !71, metadata ptr %b.addr, metadata !DIExpression()), !dbg !27 ; VAR:b
  store i32 9, ptr %e.addr, !DIAssignID !78 ; VAR:e
  call void @llvm.dbg.assign(metadata i32 9, metadata !75, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !78, metadata ptr %e.addr, metadata !DIExpression()), !dbg !27 ; VAR:e
  store i32 3, ptr %a.addr, !DIAssignID !63 ; VAR:a
  store i32 4, ptr %b.addr, !DIAssignID !65 ; VAR:b
  store i64 5, ptr %c.addr, !DIAssignID !69 ; VAR:c
  call void @llvm.dbg.assign(metadata i64 5, metadata !67, metadata !DIExpression(), metadata !69, metadata ptr %c.addr, metadata !DIExpression()), !dbg !27 ; VAR:c
  store i64 6, ptr %d.addr, !DIAssignID !74 ; VAR:d
  call void @llvm.dbg.assign(metadata i64 6, metadata !72, metadata !DIExpression(), metadata !74, metadata ptr %d.addr, metadata !DIExpression()), !dbg !27 ; VAR:d
  store i64 16, ptr %e.addr, !DIAssignID !77 ; VAR:e
  ;store i32 11, ptr %f.addr, !DIAssignID !81 ; VAR:f
  ;call void @llvm.dbg.assign(metadata i32 11, metadata !79, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !81, metadata ptr %f.addr, metadata !DIExpression()), !dbg !27 ; VAR:f
  br label %do.body, !dbg !24
; CHECK-LABEL: bb.0.entry:
; CHECK-NEXT: successors
; CHECK-NEXT: {{^ *$}}
; CHECK-NEXT:    DBG_VALUE %stack.0.a.addr, $noreg, ![[a]], !DIExpression(DW_OP_deref)
; CHECK-NEXT:    DBG_VALUE %stack.1.b.addr, $noreg, ![[b]], !DIExpression(DW_OP_deref)
; CHECK-NEXT:    DBG_VALUE %stack.4.e.addr, $noreg, ![[e]], !DIExpression(DW_OP_deref)
; CHECK-NEXT:    MOV64mi32 %stack.0.a.addr, 1, $noreg, 0, $noreg, 1
; CHECK-NEXT:    MOV64mi32 %stack.1.b.addr, 1, $noreg, 0, $noreg, 2
; CHECK-NEXT:    MOV32mi %stack.0.a.addr, 1, $noreg, 0, $noreg, 3
; CHECK-NEXT:    DBG_VALUE $noreg, $noreg, ![[a]], !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; CHECK-NEXT:    DBG_VALUE %stack.0.a.addr, $noreg, ![[a]], !DIExpression(DW_OP_plus_uconst, 4, DW_OP_deref, DW_OP_LLVM_fragment, 32, 32)
; CHECK-NEXT:    MOV32mi %stack.1.b.addr, 1, $noreg, 0, $noreg, 4
; CHECK-NEXT:    DBG_VALUE $noreg, $noreg, ![[b]], !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; CHECK-NEXT:    DBG_VALUE %stack.1.b.addr, $noreg, ![[b]], !DIExpression(DW_OP_plus_uconst, 4, DW_OP_deref, DW_OP_LLVM_fragment, 32, 32)
; CHECK-NEXT:    MOV64mi32 %stack.2.c.addr, 1, $noreg, 0, $noreg, 5
; CHECK-NEXT:    MOV64mi32 %stack.3.d.addr, 1, $noreg, 0, $noreg, 6
; CHECK-NEXT:    MOV64mi32 %stack.4.e.addr, 1, $noreg, 0, $noreg, 16
; CHECK-NEXT:    DBG_VALUE $noreg, $noreg, ![[e]], !DIExpression()
; CHECK-NEXT: {{^ *$}}

do.body:                                          ; preds = %do.cond4, %entry
  call void @llvm.dbg.assign(metadata i64 16, metadata !75, metadata !DIExpression(), metadata !77, metadata ptr %e.addr, metadata !DIExpression()), !dbg !27 ; VAR:e
  ;call void @llvm.dbg.assign(metadata i32 11, metadata !79, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !81, metadata ptr %f.addr, metadata !DIExpression()), !dbg !27 ; VAR:f
  %.pre10 = load i32, ptr @g_a, align 4, !dbg !27
  br label %do.body1, !dbg !34
; CHECK-LABEL: bb.1.do.body:
; CHECK-NEXT: successors
; CHECK-NEXT: {{^ *$}}
; CHECK-NEXT:    DBG_VALUE 16, $noreg, ![[e]], !DIExpression()
; CHECK-NEXT:    %0:gr32 = MOV32rm $rip, 1, $noreg, @g_a, $noreg
; CHECK-NEXT: {{^ *$}}

do.body1:                                         ; preds = %do.cond, %do.body
  %0 = phi i32 [ %.pre10, %do.body ], [ %1, %do.cond ], !dbg !27
  ;call void @llvm.dbg.assign(metadata i32 11, metadata !79, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !81, metadata ptr %f.addr, metadata !DIExpression()), !dbg !27 ; VAR:f
  %tobool.not = icmp eq i32 %0, 0, !dbg !27
  br i1 %tobool.not, label %if.else, label %if.then, !dbg !35
; CHECK-LABEL: bb.2.do.body1:
; CHECK-NEXT: successors
; CHECK-NEXT: {{^ *$}}
; CHECK:         JMP_1
; CHECK-NEXT: {{^ *$}}

if.then:                                          ; preds = %do.body1
  %.pre = load i32, ptr @g_a, align 4, !dbg !27
  store i32 %.pre, ptr %b.addr, !DIAssignID !66 ; VAR:b
  call void @llvm.dbg.assign(metadata i32 %.pre, metadata !22, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !66, metadata ptr %b.addr, metadata !DIExpression()), !dbg !27 ; VAR:b
  store i32 6, ptr %d.addr, !DIAssignID !74 ; VAR:d
  call void @llvm.dbg.assign(metadata i32 6, metadata !72, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata !74, metadata ptr %d.addr, metadata !DIExpression()), !dbg !27 ; VAR:d
  call void @llvm.dbg.assign(metadata i32 8, metadata !75, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata !82, metadata ptr %e.addr, metadata !DIExpression()), !dbg !27 ; VAR:e
  store i32 8, ptr %e.addr, !DIAssignID !82 ; VAR:e
  br label %do.cond, !dbg !39
; CHECK-LABEL: bb.3.if.then:
; CHECK-NEXT: successors
; CHECK-NEXT: {{^ *$}}
; CHECK-NEXT:    %5:gr32 = MOV32rm $rip, 1, $noreg, @g_a, $noreg
; CHECK-NEXT:    MOV32mr %stack.1.b.addr, 1, $noreg, 0, $noreg, killed %5
; CHECK-NEXT:    DBG_VALUE %stack.1.b.addr, $noreg, ![[b]], !DIExpression(DW_OP_deref)
; CHECK-NEXT:    MOV32mi %stack.3.d.addr, 1, $noreg, 0, $noreg, 6
; CHECK-NEXT:    DBG_VALUE 8, $noreg, ![[e]], !DIExpression(DW_OP_LLVM_fragment, 32, 32)
; CHECK-NEXT:    MOV32mi %stack.4.e.addr, 1, $noreg, 0, $noreg, 8
; CHECK-NEXT:    DBG_VALUE %stack.4.e.addr, $noreg, ![[e]], !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 32, 32)
; CHECK-NEXT:    JMP_1 %bb.5

if.else:                                          ; preds = %do.body1
  store i32 6, ptr %d.addr, !DIAssignID !74 ; VAR:d
  call void @llvm.dbg.assign(metadata i32 6, metadata !72, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !74, metadata ptr %d.addr, metadata !DIExpression()), !dbg !27 ; VAR:d
  %e.high32 = getelementptr i32, ptr %e.addr, i32 1
  store i32 15, ptr %e.high32 ; VAR:e
  ;store i32 10, ptr %f.addr ; VAR:f
  br label %do.cond
; CHECK-LABEL: bb.4.if.else:
; CHECK-NEXT: successors
; CHECK-NEXT: {{^ *$}}
; CHECK-NEXT:    MOV32mi %stack.3.d.addr, 1, $noreg, 0, $noreg, 6
; CHECK-NEXT:    MOV32mi %stack.4.e.addr, 1, $noreg, 4, $noreg, 15
; CHECK-NEXT:    DBG_VALUE %stack.4.e.addr, $noreg, ![[e]], !DIExpression(DW_OP_plus_uconst, 4, DW_OP_deref, DW_OP_LLVM_fragment, 32, 32)
; CHECK: {{^ *$}}

do.cond:                                          ; preds = %if.then, %if.else
  call void @llvm.dbg.assign(metadata i1 undef, metadata !67, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !69, metadata ptr %c.addr, metadata !DIExpression()), !dbg !27 ; VAR:c
  call void @llvm.dbg.assign(metadata i1 undef, metadata !67, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata !69, metadata ptr %c.addr, metadata !DIExpression()), !dbg !27 ; VAR:c
  ;call void @llvm.dbg.assign(metadata i32 11, metadata !79, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !81, metadata ptr %f.addr, metadata !DIExpression()), !dbg !27 ; VAR:f
  %1 = load i32, ptr @g_b, align 4, !dbg !43
  %tobool3.not = icmp eq i32 %1, 0, !dbg !43
  br i1 %tobool3.not, label %do.cond4, label %do.body1, !dbg !44, !llvm.loop !45
; CHECK-LABEL: bb.5.do.cond:
; CHECK-NEXT: successors
; CHECK-NEXT: {{^ *$}}
; CHECK-NOT: DBG_VALUE
; CHECK:      {{^ *$}}

do.cond4:                                         ; preds = %do.cond
  %2 = load i32, ptr @g_c, align 4, !dbg !48
  %tobool5.not = icmp eq i32 %2, 0, !dbg !48
  br i1 %tobool5.not, label %do.end6, label %do.body, !dbg !49, !llvm.loop !50
; CHECK-LABEL: bb.6.do.cond4:
; CHECK-NEXT: successors
; CHECK-NEXT: {{^ *$}}
; CHECK-NOT:     DBG
; CHECK: {{^ *$}}

do.end6:                                          ; preds = %do.cond4
  call void @llvm.dbg.assign(metadata i32 3, metadata !21, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !63, metadata ptr %a.addr, metadata !DIExpression()), !dbg !27; VAR:a
  call void @llvm.dbg.assign(metadata i32 0, metadata !21, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata !70, metadata ptr %a.addr, metadata !DIExpression(DW_OP_plus_uconst, 32)), !dbg !27; VAR:a
  ;call void @llvm.dbg.assign(metadata i32 3, metadata !21, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata !81, metadata ptr %a.addr, metadata !DIExpression()), !dbg !27; VAR:a
  call void @llvm.dbg.assign(metadata i32 4, metadata !22, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !65, metadata ptr %b.addr, metadata !DIExpression()), !dbg !27 ; VAR:b
  call void @llvm.dbg.assign(metadata i32 0, metadata !22, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata !71, metadata ptr %b.addr, metadata !DIExpression()), !dbg !27 ; VAR:b
  call void @llvm.dbg.assign(metadata i1 undef, metadata !67, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !69, metadata ptr %c.addr, metadata !DIExpression()), !dbg !27 ; VAR:c
  call void @llvm.dbg.assign(metadata i1 undef, metadata !67, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata !69, metadata ptr %c.addr, metadata !DIExpression()), !dbg !27 ; VAR:c
  call void @llvm.dbg.assign(metadata i32 6, metadata !72, metadata !DIExpression(), metadata !74, metadata ptr %d.addr, metadata !DIExpression()), !dbg !27 ; VAR:d
  ;call void @llvm.dbg.assign(metadata i32 11, metadata !79, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !81, metadata ptr %f.addr, metadata !DIExpression()), !dbg !27 ; VAR:f
  ret i32 0, !dbg !53
; CHECK-LABEL: bb.7.do.end6:
; CHECK-NEXT:    DBG_VALUE %stack.0.a.addr, $noreg, ![[a]], !DIExpression(DW_OP_plus_uconst, 32, DW_OP_deref, DW_OP_LLVM_fragment, 32, 32)
; CHECK-NEXT:    DBG_VALUE %stack.0.a.addr, $noreg, ![[a]], !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 0, 32)
; CHECK-NEXT:    DBG_VALUE 4, $noreg, ![[b]], !DIExpression(DW_OP_LLVM_fragment, 0, 32)
; CHECK-NEXT:    DBG_VALUE %stack.1.b.addr, $noreg, ![[b]], !DIExpression(DW_OP_deref, DW_OP_LLVM_fragment, 32, 32)
}

declare !dbg !54 void @_Z4calli(i32 noundef) local_unnamed_addr #1
declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15, !1000}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g_a", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 16.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{!0, !5, !8}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "g_b", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "long long", size: 64, encoding: DW_ATE_signed)
!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = distinct !DIGlobalVariable(name: "g_c", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!10 = !{i32 7, !"Dwarf Version", i32 5}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 8, !"PIC Level", i32 2}
!14 = !{i32 7, !"PIE Level", i32 2}
!15 = !{i32 7, !"uwtable", i32 2}
!16 = !{!"clang version 16.0.0"}
!17 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funii", scope: !3, file: !3, line: 3, type: !18, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !20)
!18 = !DISubroutineType(types: !19)
!19 = !{!7, !7, !7}
!20 = !{!21, !22}
!21 = !DILocalVariable(name: "a", arg: 1, scope: !17, file: !3, line: 3, type: !7)
!22 = !DILocalVariable(name: "b", arg: 2, scope: !17, file: !3, line: 3, type: !7)
!23 = !DILocation(line: 0, scope: !17)
!24 = !DILocation(line: 4, column: 3, scope: !17)
!25 = !DILocation(line: 5, column: 5, scope: !26)
!26 = distinct !DILexicalBlock(scope: !17, file: !3, line: 4, column: 6)
!27 = !DILocation(line: 7, column: 11, scope: !28)
!28 = distinct !DILexicalBlock(scope: !29, file: !3, line: 7, column: 11)
!29 = distinct !DILexicalBlock(scope: !26, file: !3, line: 6, column: 8)
!34 = !DILocation(line: 6, column: 5, scope: !26)
!35 = !DILocation(line: 7, column: 11, scope: !29)
!36 = !DILocation(line: 8, column: 11, scope: !37)
!37 = distinct !DILexicalBlock(scope: !28, file: !3, line: 7, column: 16)
!38 = !DILocation(line: 9, column: 9, scope: !37)
!39 = !DILocation(line: 10, column: 7, scope: !37)
!40 = !DILocation(line: 11, column: 11, scope: !41)
!41 = distinct !DILexicalBlock(scope: !28, file: !3, line: 10, column: 14)
!42 = !DILocation(line: 0, scope: !28)
!43 = !DILocation(line: 13, column: 14, scope: !26)
!44 = !DILocation(line: 13, column: 5, scope: !29)
!45 = distinct !{!45, !34, !46, !47}
!46 = !DILocation(line: 13, column: 17, scope: !26)
!47 = !{!"llvm.loop.mustprogress"}
!48 = !DILocation(line: 14, column: 12, scope: !17)
!49 = !DILocation(line: 14, column: 3, scope: !26)
!50 = distinct !{!50, !24, !51, !47}
!51 = !DILocation(line: 14, column: 15, scope: !17)
!52 = !DILocation(line: 15, column: 12, scope: !17)
!53 = !DILocation(line: 15, column: 3, scope: !17)
!54 = !DISubprogram(name: "call", linkageName: "_Z4calli", scope: !3, file: !3, line: 2, type: !55, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !57)
!55 = !DISubroutineType(types: !56)
!56 = !{null, !7}
!57 = !{}
!58 = distinct !DIAssignID()
!63 = distinct !DIAssignID()
!64 = distinct !DIAssignID()
!65 = distinct !DIAssignID()
!66 = distinct !DIAssignID()
!67 = !DILocalVariable(name: "c", scope: !17, file: !3, line: 3, type: !7)
!68 = distinct !DIAssignID()
!69 = distinct !DIAssignID()
!70 = distinct !DIAssignID()
!71 = distinct !DIAssignID()
!72 = !DILocalVariable(name: "d", scope: !17, file: !3, line: 3, type: !7)
!73 = distinct !DIAssignID()
!74 = distinct !DIAssignID()
!75 = !DILocalVariable(name: "e", scope: !17, file: !3, line: 3, type: !7)
!76 = distinct !DIAssignID()
!77 = distinct !DIAssignID()
!78 = distinct !DIAssignID()
!82 = distinct !DIAssignID()
!1000 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
