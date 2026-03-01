; RUN: llc %s -stop-after=finalize-isel -o - \
; RUN:   | FileCheck %s

;; Test that assignment tracking correctly handles calls where a pointer to a
;; tracked alloca escapes as an argument. After such a call, the memory
;; location should be reinstated because the callee may have modified the
;; variable through the pointer.
;;
;; Each function uses a #dbg_value to force LocKind::Val at some point, which
;; prevents the variable from being "always stack homed" and causes the
;; analysis to emit per-instruction DBG_VALUE records.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @clobber(ptr)
declare i32 @readonly_func(ptr readonly)
declare void @byval_func(ptr byval(i32))

;;Test 1: Basic escaping call reinstates memory location
;;
;; After the #dbg_value switches to Val (DBG_VALUE $noreg because %a has no
;; vreg), the escaping call to @clobber should reinstate the memory location.
;;
; CHECK-LABEL: name: test_basic_escaping_call
; CHECK:       bb.0.entry:
; CHECK:         DBG_VALUE %stack.0.x, $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_deref)
; CHECK:         MOV32mi %stack.0.x
; CHECK:         DBG_VALUE $noreg, $noreg, !{{[0-9]+}}, !DIExpression()
; CHECK:         CALL64pcrel32 {{.*}}@clobber
;; After the escaping call, memory location is reinstated:
; CHECK:         DBG_VALUE %stack.0.x, $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_deref)
; CHECK:         RET 0

define void @test_basic_escaping_call(i32 %a) !dbg !7 {
entry:
  %x = alloca i32, align 4, !DIAssignID !20
    #dbg_assign(i1 poison, !11, !DIExpression(), !20, ptr %x, !DIExpression(), !12)
  store i32 1, ptr %x, align 4, !DIAssignID !21
    #dbg_assign(i32 1, !11, !DIExpression(), !21, ptr %x, !DIExpression(), !12)
  #dbg_value(i32 %a, !11, !DIExpression(), !12)
  call void @clobber(ptr %x)
  ret void, !dbg !13
}

;;Test 2: Escaping call followed by a tagged store
;;
;; Verifies that the escaping call resets state so the subsequent tagged
;; store correctly shows Mem (no stale value from before the call).
;;
; CHECK-LABEL: name: test_escaping_then_store
; CHECK:       bb.0.entry:
; CHECK:         DBG_VALUE %stack.0.y, $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_deref)
; CHECK:         MOV32mi %stack.0.y
; CHECK:         DBG_VALUE $noreg, $noreg, !{{[0-9]+}}, !DIExpression()
; CHECK:         CALL64pcrel32 {{.*}}@clobber
;; After escaping call, memory location reinstated:
; CHECK:         DBG_VALUE %stack.0.y, $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_deref)
;; Then the second store (still Mem, redundant DBG_VALUE elided):
; CHECK:         MOV32mi %stack.0.y
; CHECK:         RET 0

define void @test_escaping_then_store(i32 %a) !dbg !30 {
entry:
  %y = alloca i32, align 4, !DIAssignID !40
    #dbg_assign(i1 poison, !31, !DIExpression(), !40, ptr %y, !DIExpression(), !32)
  store i32 1, ptr %y, align 4, !DIAssignID !41
    #dbg_assign(i32 1, !31, !DIExpression(), !41, ptr %y, !DIExpression(), !32)
  #dbg_value(i32 %a, !31, !DIExpression(), !32)
  call void @clobber(ptr %y)
  store i32 2, ptr %y, align 4, !DIAssignID !42
    #dbg_assign(i32 2, !31, !DIExpression(), !42, ptr %y, !DIExpression(), !32)
  ret void, !dbg !33
}

;;Test 3: Readonly call should NOT reinstate memory location
;;
;; A readonly call cannot modify memory, so no DBG_VALUE after the call.
;;
; CHECK-LABEL: name: test_readonly_not_escaping
; CHECK:       bb.0.entry:
; CHECK:         DBG_VALUE %stack.0.z, $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_deref)
; CHECK:         MOV32mi %stack.0.z
; CHECK:         DBG_VALUE $noreg, $noreg, !{{[0-9]+}}, !DIExpression()
; CHECK:         CALL64pcrel32 {{.*}}@readonly_func
; CHECK-NOT:     DBG_VALUE
; CHECK:         RET 0

define void @test_readonly_not_escaping(i32 %a) !dbg !50 {
entry:
  %z = alloca i32, align 4, !DIAssignID !60
    #dbg_assign(i1 poison, !51, !DIExpression(), !60, ptr %z, !DIExpression(), !52)
  store i32 42, ptr %z, align 4, !DIAssignID !61
    #dbg_assign(i32 42, !51, !DIExpression(), !61, ptr %z, !DIExpression(), !52)
  #dbg_value(i32 %a, !51, !DIExpression(), !52)
  %r = call i32 @readonly_func(ptr readonly %z)
  ret void, !dbg !53
}

;;Test 4: Byval call should NOT reinstate memory location
;;
;; A byval argument passes a copy. The callee cannot modify the original.
;;
; CHECK-LABEL: name: test_byval_not_escaping
; CHECK:       bb.0.entry:
; CHECK:         DBG_VALUE %stack.0.w, $noreg, !{{[0-9]+}}, !DIExpression(DW_OP_deref)
; CHECK:         MOV32mi %stack.0.w
; CHECK:         DBG_VALUE $noreg, $noreg, !{{[0-9]+}}, !DIExpression()
; CHECK:         CALL64pcrel32 {{.*}}@byval_func
; CHECK-NOT:     DBG_VALUE
; CHECK:         RET 0

define void @test_byval_not_escaping(i32 %a) !dbg !70 {
entry:
  %w = alloca i32, align 4, !DIAssignID !80
    #dbg_assign(i1 poison, !71, !DIExpression(), !80, ptr %w, !DIExpression(), !72)
  store i32 10, ptr %w, align 4, !DIAssignID !81
    #dbg_assign(i32 10, !71, !DIExpression(), !81, ptr %w, !DIExpression(), !72)
  #dbg_value(i32 %a, !71, !DIExpression(), !72)
  call void @byval_func(ptr byval(i32) %w)
  ret void, !dbg !73
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

; Function 1 metadata
!6 = !DISubroutineType(types: !2)
!7 = distinct !DISubprogram(name: "test_basic_escaping_call", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DILocalVariable(name: "x", scope: !7, file: !1, line: 2, type: !14)
!12 = !DILocation(line: 2, column: 1, scope: !7)
!13 = !DILocation(line: 5, column: 1, scope: !7)
!20 = distinct !DIAssignID()
!21 = distinct !DIAssignID()

; Function 2 metadata
!30 = distinct !DISubprogram(name: "test_escaping_then_store", scope: !1, file: !1, line: 10, type: !6, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!31 = !DILocalVariable(name: "y", scope: !30, file: !1, line: 11, type: !14)
!32 = !DILocation(line: 11, column: 1, scope: !30)
!33 = !DILocation(line: 15, column: 1, scope: !30)
!40 = distinct !DIAssignID()
!41 = distinct !DIAssignID()
!42 = distinct !DIAssignID()

; Function 3 metadata
!50 = distinct !DISubprogram(name: "test_readonly_not_escaping", scope: !1, file: !1, line: 20, type: !6, scopeLine: 20, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!51 = !DILocalVariable(name: "z", scope: !50, file: !1, line: 21, type: !14)
!52 = !DILocation(line: 21, column: 1, scope: !50)
!53 = !DILocation(line: 25, column: 1, scope: !50)
!60 = distinct !DIAssignID()
!61 = distinct !DIAssignID()

; Function 4 metadata
!70 = distinct !DISubprogram(name: "test_byval_not_escaping", scope: !1, file: !1, line: 30, type: !6, scopeLine: 30, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!71 = !DILocalVariable(name: "w", scope: !70, file: !1, line: 31, type: !14)
!72 = !DILocation(line: 31, column: 1, scope: !70)
!73 = !DILocation(line: 35, column: 1, scope: !70)
!80 = distinct !DIAssignID()
!81 = distinct !DIAssignID()