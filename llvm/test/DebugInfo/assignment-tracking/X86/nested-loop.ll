; RUN: llc %s -stop-after=finalize-isel -o - -experimental-assignment-tracking \
; RUN: | FileCheck %s --implicit-check-not=DBG

;; Test a variety of block inputs and lattice configurations for the assignment
;; tracking analysis (debug-ata).

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

;;  Key
;; ╔═════════════════════╦═══════════════════════════════════════════════════════════════════╗
;; ║ thing               ║ meaning                                                           ║
;; ╠═════════════════════╬═══════════════════════════════════════════════════════════════════╣
;; ║ mem=<!n|phi>        ║ assignment of !n or phi to memory                                 ║
;; ║ dbg=<!n|phi>        ║ assignment of !n or phi to source variable                        ║
;; ║ phi                 ║ phi of assignments (operands not traked)*                         ║
;; ║ loc=<val|mem|none>  ║ location to use is value (implicit location), stack home, or none ║
;; ╚═════════════════════╩═══════════════════════════════════════════════════════════════════╝
;;  (*) A phi in the def column represents an assignment made by an untagged store.
;;
;; Variable 'a' (!21)
;; Check initial dbg and mem assignment values are propagated through all blocks.
;; ╔═════════════╦══════════════════════════╦═════════════════╦══════════════════════════╗
;; ║ block       ║ in                       ║ def             ║ out                      ║
;; ╠═════════════╬══════════════════════════╬═════════════════╬══════════════════════════╣
;; ║ entry       ║                          ║ mem=!63 dbg=!70 ║ mem=!63 dbg=!70 loc=val  ║
;; ║ do.end6     ║ mem=!63 dbg=!70 loc=val  ║ mem=!63         ║ mem=!63 dbg=!63 loc=mem  ║
;; ╚═════════════╩══════════════════════════╩═════════════════╩══════════════════════════╝
;;
;; Variable 'b' (!22)
;; Check mem=dbg assignment on branch in nested loop causes a mem=phi (tested by looking
;; for value-based DBG_VALUE in do.end6).
;; ╔═════════════╦══════════════════════════╦═════════════════╦══════════════════════════╗
;; ║ block       ║ in                       ║ def             ║ out                      ║
;; ╠═════════════╬══════════════════════════╬═════════════════╬══════════════════════════╣
;; ║ entry       ║                          ║ mem=!65 dbg=!71 ║ mem=!65 dbg=!71 loc=val  ║
;; ║ if.then     ║ mem=phi dbg=phi loc=none ║ mem=!66 dbg=!66 ║ mem=!66 dbg=!66 loc=mem  ║
;; ║ do.end6     ║ mem=phi dbg=phi loc=none ║ mem=!65         ║ mem=phi dbg=!65 loc=val  ║
;; ╚═════════════╩══════════════════════════╩═════════════════╩══════════════════════════╝
;;
;; Variable 'c' (!67)
;; Check initial dbg and mem assignment values are propagated through all blocks, with
;; dbg defs with the inital assignment ID put in do.cond and do.end6 (variable is always
;; in memory).
;; ╔═════════════╦══════════════════════════╦═════════════════╦══════════════════════════╗
;; ║ block       ║ in                       ║ def             ║ out                      ║
;; ╠═════════════╬══════════════════════════╬═════════════════╬══════════════════════════╣
;; ║ entry       ║                          ║ mem=!69 dbg=!69 ║ mem=!69 dbg=!69 loc=mem  ║
;; ║ do.cond     ║ mem=!69 dbg=!69 loc=mem  ║         dbg=!69 ║ mem=!69 dbg=!69 loc=mem  ║
;; ║ do.end6     ║ mem=!69 dbg=!69 loc=mem  ║         dbg=!69 ║ mem=!69 dbg=!69 loc=mem  ║
;; ╚═════════════╩══════════════════════════╩═════════════════╩══════════════════════════╝
;;
;; Variable 'd' (!72)
;; Same as above, except the dbg def in do.cond has been swapped for a dbg=mem def (with
;; the initial assignment ID) and has been moved to if.else.
;; ╔═════════════╦══════════════════════════╦═════════════════╦══════════════════════════╗
;; ║ block       ║ in                       ║ def             ║ out                      ║
;; ╠═════════════╬══════════════════════════╬═════════════════╬══════════════════════════╣
;; ║ entry       ║                          ║ mem=!74 dbg=!74 ║ mem=!74 dbg=!74 loc=mem  ║
;; ║ if.else     ║ mem=!74 dbg=!74 loc=mem  ║ mem=!74 dbg=!74 ║ mem=!74 dbg=!74 loc=mem  ║
;; ║ do.end6     ║ mem=!74 dbg=!74 loc=mem  ║         dbg=!74 ║ mem=!74 dbg=!74 loc=mem  ║
;; ╚═════════════╩══════════════════════════╩═════════════════╩══════════════════════════╝
;;
;; Variable 'e' (!75)
;; mem defs in entry, if.then and if.else with same ID (!77). Check these join correct
;; (tested using the dbg defs of the same ID - the memory location is valid at each of
;; these with that ID).
;; ╔═════════════╦══════════════════════════╦═════════════════╦══════════════════════════╗
;; ║ block       ║ in                       ║ def             ║ out                      ║
;; ╠═════════════╬══════════════════════════╬═════════════════╬══════════════════════════╣
;; ║ entry       ║                          ║ mem=!77 dbg=!78 ║ mem=!77 dbg=!78 loc=val  ║
;; ║ do.body     ║ mem=!77 dbg=phi loc=none ║         dbg=!77 ║ mem=!77 dbg=!77 loc=mem  ║
;; ║ do.body1    ║ mem=!77 dbg=!77 loc=mem  ║         dbg=!77 ║ mem=!77 dbg=!77 loc=mem  ║
;; ║ if.then     ║ mem=!77 dbg=!77 loc=mem  ║ mem=!77         ║ mem=!77 dbg=!77 loc=mem  ║
;; ║ if.else     ║ mem=!77 dbg=!77 loc=mem  ║ mem=!77         ║ mem=!77 dbg=!77 loc=mem  ║
;; ╚═════════════╩══════════════════════════╩═════════════════╩══════════════════════════╝
;;
;; Variable 'f' (!79)
;; mem def in entry and an untagged store in if.else (results in mem=phi, dbg=phi defs).
;; Use dbg defs in do.body, do.body1, do.cond and do.end6 to check the phi-ness
;; has been propagated (the memory loc at each is not a valid location). Check the memory
;; loc is used in if.else after the untagged store.
;; ╔═════════════╦══════════════════════════╦═════════════════╦══════════════════════════╗
;; ║ block       ║ in                       ║ def             ║ out                      ║
;; ╠═════════════╬══════════════════════════╬═════════════════╬══════════════════════════╣
;; ║ entry       ║                          ║ mem=!81 dbg=!81 ║ mem=!81 dbg=!81 loc=mem  ║
;; ║ do.body     ║ mem=phi dbg=phi loc=none ║         dbg=!81 ║ mem=phi dbg=!81 loc=val  ║
;; ║ do.body1    ║ mem=phi dbg=phi loc=none ║         dbg=!81 ║ mem=phi dbg=!81 loc=val  ║
;; ║ if.else     ║ mem=phi dbg=phi loc=none ║ mem=phi dbg=phi ║ mem=phi dbg=phi loc=mem  ║
;; ║ do.cond     ║ mem=phi dbg=phi loc=none ║         dbg=!81 ║ mem=phi dbg=!81 loc=val  ║
;; ║ do.end6     ║ mem=phi dbg=!81 loc=val  ║         dbg=!81 ║ mem=!69 dbg=!81 loc=val  ║
;; ╚═════════════╩══════════════════════════╩═════════════════╩══════════════════════════╝
;;
;; Variable 'g' (!82)
;; Check that joining loc=none with anything else results in loc=none. The out-loc of
;; entry is set up to be loc=none by following an untagged store with a tagged store,
;; with the linked dbg.assign in another block. The dbg.assign is in do.body - it follows
;; another store linked to it. Importantly, there are other instructions wedged between
;; them, which is how we test that the in-loc is loc=none. The result of encountering
;; a tagged store while the loc=none is to emit nothing. Thus, we check that no location
;; def is emitted in do.body until the dbg.assign is encountered (after the load that was
;; wedged between the store and intrinsic).
;; ╔═════════════╦══════════════════════════╦═════════════════╦══════════════════════════╗
;; ║ block       ║ in                       ║ def             ║ out                      ║
;; ╠═════════════╬══════════════════════════╬═════════════════╬══════════════════════════╣
;; ║ entry       ║                          ║ mem=phi dbg=phi ║ mem=phi dbg=phi loc=none ║
;; ║ do.body     ║ mem=phi dbg=phi loc=none ║ mem=!84 dbg=!84 ║ mem=!84 dbg=!84 loc=mem  ║
;; ╚═════════════╩══════════════════════════╩═════════════════╩══════════════════════════╝

; CHECK-DAG: ![[a:[0-9]+]] = !DILocalVariable(name: "a",
; CHECK-DAG: ![[b:[0-9]+]] = !DILocalVariable(name: "b",
; CHECK-DAG: ![[c:[0-9]+]] = !DILocalVariable(name: "c",
; CHECK-DAG: ![[d:[0-9]+]] = !DILocalVariable(name: "d",
; CHECK-DAG: ![[e:[0-9]+]] = !DILocalVariable(name: "e",
; CHECK-DAG: ![[f:[0-9]+]] = !DILocalVariable(name: "f",
; CHECK-DAG: ![[g:[0-9]+]] = !DILocalVariable(name: "g",

;; Variables 'c' (!67) and 'd' (!72) are always stack-homed.
; CHECK:      - { id: 2, name: c.addr, type: default, offset: 0, size: 4, alignment: 4,
; CHECK-NEXT:     stack-id: default, callee-saved-register: '', callee-saved-restored: true,
; CHECK-NEXT:     debug-info-variable: '![[c]]', debug-info-expression: '!DIExpression()',
; CHECK:      - { id: 3, name: d.addr, type: default, offset: 0, size: 4, alignment: 4,
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
  %a.addr = alloca i32, align 4, !DIAssignID !58 ; VAR:a
  call void @llvm.dbg.assign(metadata i1 undef, metadata !21, metadata !DIExpression(), metadata !58, metadata ptr %a.addr, metadata !DIExpression()), !dbg !27 ; VAR:a
  %b.addr = alloca i32, align 4, !DIAssignID !64 ; VAR:b
  call void @llvm.dbg.assign(metadata i1 undef, metadata !22, metadata !DIExpression(), metadata !64, metadata ptr %b.addr, metadata !DIExpression()), !dbg !27 ; VAR:b
  %c.addr = alloca i32, align 4, !DIAssignID !68 ; VAR:c
  call void @llvm.dbg.assign(metadata i1 undef, metadata !67, metadata !DIExpression(), metadata !68, metadata ptr %c.addr, metadata !DIExpression()), !dbg !27 ; VAR:c
  %d.addr = alloca i32, align 4, !DIAssignID !73 ; VAR:d
  call void @llvm.dbg.assign(metadata i1 undef, metadata !72, metadata !DIExpression(), metadata !73, metadata ptr %d.addr, metadata !DIExpression()), !dbg !27 ; VAR:d
  %e.addr = alloca i32, align 4, !DIAssignID !76 ; VAR:e
  call void @llvm.dbg.assign(metadata i1 undef, metadata !75, metadata !DIExpression(), metadata !76, metadata ptr %e.addr, metadata !DIExpression()), !dbg !27 ; VAR:e
  %f.addr = alloca i32, align 4, !DIAssignID !80 ; VAR:f
  call void @llvm.dbg.assign(metadata i1 undef, metadata !79, metadata !DIExpression(), metadata !80, metadata ptr %f.addr, metadata !DIExpression()), !dbg !27 ; VAR:f
  %g.addr = alloca i32, align 4, !DIAssignID !83 ; VAR:g
  call void @llvm.dbg.assign(metadata i1 undef, metadata !82, metadata !DIExpression(), metadata !83, metadata ptr %g.addr, metadata !DIExpression()), !dbg !27 ; VAR:g
  store i32 1, ptr %a.addr, !DIAssignID !70 ; VAR:a
  call void @llvm.dbg.assign(metadata i32 1, metadata !21, metadata !DIExpression(), metadata !70, metadata ptr %a.addr, metadata !DIExpression()), !dbg !27 ; VAR:a
  store i32 2, ptr %b.addr, !DIAssignID !71 ; VAR:b
  call void @llvm.dbg.assign(metadata i32 2, metadata !22, metadata !DIExpression(), metadata !71, metadata ptr %b.addr, metadata !DIExpression()), !dbg !27 ; VAR:b
  store i32 12, ptr %g.addr ; VAR:g
  store i32 9, ptr %e.addr, !DIAssignID !78 ; VAR:e
  call void @llvm.dbg.assign(metadata i32 9, metadata !75, metadata !DIExpression(), metadata !78, metadata ptr %e.addr, metadata !DIExpression()), !dbg !27 ; VAR:c
  store i32 3, ptr %a.addr, !DIAssignID !63 ; VAR:a
  store i32 4, ptr %b.addr, !DIAssignID !65 ; VAR:b
  store i32 5, ptr %c.addr, !DIAssignID !69 ; VAR:c
  call void @llvm.dbg.assign(metadata i32 5, metadata !67, metadata !DIExpression(), metadata !69, metadata ptr %c.addr, metadata !DIExpression()), !dbg !27 ; VAR:c
  store i32 6, ptr %d.addr, !DIAssignID !74 ; VAR:d
  call void @llvm.dbg.assign(metadata i32 6, metadata !72, metadata !DIExpression(), metadata !74, metadata ptr %d.addr, metadata !DIExpression()), !dbg !27 ; VAR:d
  store i32 8, ptr %e.addr, !DIAssignID !77 ; VAR:e
  store i32 13, ptr %g.addr, !DIAssignID !84 ; VAR:g
  store i32 11, ptr %f.addr, !DIAssignID !81 ; VAR:f
  call void @llvm.dbg.assign(metadata i32 11, metadata !79, metadata !DIExpression(), metadata !81, metadata ptr %f.addr, metadata !DIExpression()), !dbg !27 ; VAR:f
  br label %do.body, !dbg !24
; CHECK-LABEL: bb.0.entry:
; CHECK-NEXT: successors
; CHECK-NEXT: {{^ *$}}
; CHECK-NEXT:    DBG_VALUE %stack.0.a.addr, $noreg, ![[a]], !DIExpression(DW_OP_deref)
; CHECK-NEXT:    DBG_VALUE %stack.1.b.addr, $noreg, ![[b]], !DIExpression(DW_OP_deref)
; CHECK-NEXT:    DBG_VALUE %stack.4.e.addr, $noreg, ![[e]], !DIExpression(DW_OP_deref)
; CHECK-NEXT:    DBG_VALUE %stack.5.f.addr, $noreg, ![[f]], !DIExpression(DW_OP_deref)
; CHECK-NEXT:    DBG_VALUE %stack.6.g.addr, $noreg, ![[g]], !DIExpression(DW_OP_deref)
; CHECK-NEXT:    MOV32mi %stack.0.a.addr, 1, $noreg, 0, $noreg, 3
; CHECK-NEXT:    DBG_VALUE 1, $noreg, ![[a]], !DIExpression()
; CHECK-NEXT:    MOV32mi %stack.1.b.addr, 1, $noreg, 0, $noreg, 4
; CHECK-NEXT:    DBG_VALUE 2, $noreg, ![[b]], !DIExpression()
; CHECK-NEXT:    MOV32mi %stack.2.c.addr, 1, $noreg, 0, $noreg, 5
; CHECK-NEXT:    MOV32mi %stack.3.d.addr, 1, $noreg, 0, $noreg, 6
; CHECK-NEXT:    MOV32mi %stack.4.e.addr, 1, $noreg, 0, $noreg, 8
; CHECK-NEXT:    DBG_VALUE 9, $noreg, ![[e]], !DIExpression()
; CHECK-NEXT:    MOV32mi %stack.6.g.addr, 1, $noreg, 0, $noreg, 13
; CHECK-NEXT:    DBG_VALUE $noreg, $noreg, ![[g]], !DIExpression()
; CHECK-NEXT:    MOV32mi %stack.5.f.addr, 1, $noreg, 0, $noreg, 11
; CHECK-NEXT: {{^ *$}}

do.body:                                          ; preds = %do.cond4, %entry
  call void @llvm.dbg.assign(metadata i32 8, metadata !75, metadata !DIExpression(), metadata !77, metadata ptr %e.addr, metadata !DIExpression()), !dbg !27 ; VAR:e
  call void @llvm.dbg.assign(metadata i32 11, metadata !79, metadata !DIExpression(), metadata !81, metadata ptr %f.addr, metadata !DIExpression()), !dbg !27 ; VAR:f
  store i32 13, ptr %g.addr, !DIAssignID !84 ; VAR:g
  %.pre10 = load i32, ptr @g_a, align 4, !dbg !27
  call void @llvm.dbg.assign(metadata i32 11, metadata !82, metadata !DIExpression(), metadata !84, metadata ptr %g.addr, metadata !DIExpression()), !dbg !27 ; VAR:g
  br label %do.body1, !dbg !34
; CHECK-LABEL: bb.1.do.body:
; CHECK-NEXT: successors
; CHECK-NEXT: {{^ *$}}
; CHECK-NEXT:    DBG_VALUE %stack.4.e.addr, $noreg, ![[e]], !DIExpression(DW_OP_deref)
; CHECK-NEXT:    DBG_VALUE 11, $noreg, ![[f]], !DIExpression()
; CHECK-NEXT:    MOV32mi %stack.6.g.addr, 1, $noreg, 0, $noreg, 13
; CHECK-NEXT:    %0:gr32 = MOV32rm $rip, 1, $noreg, @g_a, $noreg
; CHECK-NEXT:    DBG_VALUE %stack.6.g.addr, $noreg, ![[g]], !DIExpression(DW_OP_deref)
; CHECK-NEXT: {{^ *$}}

do.body1:                                         ; preds = %do.cond, %do.body
  %0 = phi i32 [ %.pre10, %do.body ], [ %1, %do.cond ], !dbg !27
  call void @llvm.dbg.assign(metadata i32 8, metadata !75, metadata !DIExpression(), metadata !77, metadata ptr %e.addr, metadata !DIExpression()), !dbg !27 ; VAR:e
  call void @llvm.dbg.assign(metadata i32 11, metadata !79, metadata !DIExpression(), metadata !81, metadata ptr %f.addr, metadata !DIExpression()), !dbg !27 ; VAR:f
  %tobool.not = icmp eq i32 %0, 0, !dbg !27
  br i1 %tobool.not, label %if.else, label %if.then, !dbg !35
; CHECK-LABEL: bb.2.do.body1:
; CHECK-NEXT: successors
; CHECK-NEXT: {{^ *$}}
; CHECK:         DBG_VALUE %stack.4.e.addr, $noreg, ![[e]], !DIExpression(DW_OP_deref)
; CHECK-NEXT:    DBG_VALUE 11, $noreg, ![[f]], !DIExpression()
; CHECK:         JMP_1
; CHECK-NEXT: {{^ *$}}

if.then:                                          ; preds = %do.body1
  %.pre = load i32, ptr @g_a, align 4, !dbg !27
  store i32 %.pre, ptr %b.addr, !DIAssignID !66 ; VAR:b
  call void @llvm.dbg.assign(metadata i32 %.pre, metadata !22, metadata !DIExpression(), metadata !66, metadata ptr %b.addr, metadata !DIExpression()), !dbg !27 ; VAR:b
  store i32 8, ptr %e.addr, !DIAssignID !77 ; VAR:e
  br label %do.cond, !dbg !39
; CHECK-LABEL: bb.3.if.then:
; CHECK-NEXT: successors
; CHECK-NEXT: {{^ *$}}
; CHECK-NEXT:    %5:gr32 = MOV32rm
; CHECK-NEXT:    MOV32mr %stack.1.b.addr, 1, $noreg, 0, $noreg, killed %5
; CHECK-NEXT:    DBG_VALUE %stack.1.b.addr, $noreg, ![[b]], !DIExpression(DW_OP_deref
; CHECK-NEXT:    MOV32mi %stack.4.e.addr, 1, $noreg, 0, $noreg, 8
; CHECK-NEXT:    DBG_VALUE %stack.4.e.addr, $noreg, ![[e]], !DIExpression(DW_OP_deref)
; CHECK-NEXT:    JMP_1 %bb.5
; CHECK-NEXT: {{^ *$}}

if.else:                                          ; preds = %do.body1
  store i32 6, ptr %d.addr, !DIAssignID !74 ; VAR:d
  call void @llvm.dbg.assign(metadata i32 6, metadata !72, metadata !DIExpression(), metadata !74, metadata ptr %d.addr, metadata !DIExpression()), !dbg !27 ; VAR:d
  store i32 8, ptr %e.addr, !DIAssignID !77 ; VAR:e
  store i32 10, ptr %f.addr ; VAR:f
  br label %do.cond
; CHECK-LABEL: bb.4.if.else:
; CHECK-NEXT: successors
; CHECK-NEXT: {{^ *$}}
; CHECK-NEXT:    MOV32mi %stack.3.d.addr, 1, $noreg, 0, $noreg, 6
; CHECK-NEXT:    MOV32mi %stack.4.e.addr, 1, $noreg, 0, $noreg, 8
; CHECK-NEXT:    DBG_VALUE %stack.4.e.addr, $noreg, ![[e]], !DIExpression(DW_OP_deref)
; CHECK-NEXT:    MOV32mi %stack.5.f.addr, 1, $noreg, 0, $noreg, 10
; CHECK-NEXT:    DBG_VALUE %stack.5.f.addr, $noreg, ![[f]], !DIExpression(DW_OP_deref)
; CHECK-NEXT: {{^ *$}}

do.cond:                                          ; preds = %if.then, %if.else
  call void @llvm.dbg.assign(metadata i1 undef, metadata !67, metadata !DIExpression(), metadata !69, metadata ptr %c.addr, metadata !DIExpression()), !dbg !27 ; VAR:c
  call void @llvm.dbg.assign(metadata i32 11, metadata !79, metadata !DIExpression(), metadata !81, metadata ptr %f.addr, metadata !DIExpression()), !dbg !27 ; VAR:f
  %1 = load i32, ptr @g_b, align 4, !dbg !43
  %tobool3.not = icmp eq i32 %1, 0, !dbg !43
  br i1 %tobool3.not, label %do.cond4, label %do.body1, !dbg !44, !llvm.loop !45
; CHECK-LABEL: bb.5.do.cond:
; CHECK-NEXT: successors
; CHECK-NEXT: {{^ *$}}
; CHECK-NEXT:    DBG_VALUE 11, $noreg, ![[f]], !DIExpression()
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
  call void @llvm.dbg.assign(metadata i32 3, metadata !21, metadata !DIExpression(), metadata !63, metadata ptr %a.addr, metadata !DIExpression()), !dbg !27; VAR:a
  call void @llvm.dbg.assign(metadata i32 4, metadata !22, metadata !DIExpression(), metadata !65, metadata ptr %b.addr, metadata !DIExpression()), !dbg !27 ; VAR:b
  call void @llvm.dbg.assign(metadata i1 undef, metadata !67, metadata !DIExpression(), metadata !69, metadata ptr %c.addr, metadata !DIExpression()), !dbg !27 ; VAR:c
  call void @llvm.dbg.assign(metadata i32 6, metadata !72, metadata !DIExpression(), metadata !74, metadata ptr %d.addr, metadata !DIExpression()), !dbg !27 ; VAR:d
  call void @llvm.dbg.assign(metadata i32 11, metadata !79, metadata !DIExpression(), metadata !81, metadata ptr %f.addr, metadata !DIExpression()), !dbg !27 ; VAR:f
  ret i32 0, !dbg !53
; CHECK-LABEL: bb.7.do.end6:
; CHECK-NEXT:    DBG_VALUE %stack.0.a.addr, $noreg, ![[a]], !DIExpression(DW_OP_deref)
; CHECK-NEXT:    DBG_VALUE 4, $noreg, ![[b]], !DIExpression()
; CHECK-NEXT:    DBG_VALUE 11, $noreg, ![[f]], !DIExpression()
}

declare !dbg !54 void @_Z4calli(i32 noundef) local_unnamed_addr #1
declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g_a", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 16.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{!0, !5, !8}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "g_b", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
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
!59 = !DILocalVariable(name: "Arr", scope: !17, file: !3, line: 4, type: !60)
!60 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 96, elements: !61)
!61 = !{!62}
!62 = !DISubrange(count: 3)
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
!79 = !DILocalVariable(name: "f", scope: !17, file: !3, line: 3, type: !7)
!80 = distinct !DIAssignID()
!81 = distinct !DIAssignID()
!82 = !DILocalVariable(name: "g", scope: !17, file: !3, line: 3, type: !7)
!83 = distinct !DIAssignID()
!84 = distinct !DIAssignID()
