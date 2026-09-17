; NOTE: Do not autogenerate
; Debug-record coverage for static outer-epilogue fission.
;
; N denotes the reduction nest and E the post-inner epilogue.
;
; Records attach to the instruction they precede, not to the SSA value they
; describe. This multi-block positive therefore covers three nontrivial cases:
; a use-before-def record for a later E value, a record for an E value attached
; to an omitted path terminator, and a final E record attached to the last path
; terminator. A record for the N forwarding PHI shares the first E anchor but
; must remain on the reduction path.
;
; Outer-epilogue fission is opt-in, so this RUN requests it explicitly.
; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 \
; RUN:     -loop-interchange-outer-epilogue-fission \
; RUN:     -verify-dom-info -verify-loop-info -verify-scev -verify-loop-lcssa \
; RUN:     -S 2>&1 | FileCheck %s \
; RUN:     --implicit-check-not='#dbg_value({{.*}} poison'

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @multi_block_debug(ptr noalias %A, ptr noalias %D,
                               ptr noalias %EArray, ptr noalias %R) !dbg !3 {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  %chk.i = phi double [ 0.000000e+00, %entry ], [ %chk.next, %outer.latch ]
  %outer.variant = uitofp i64 %i to double
  br label %inner.header

inner.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %chk.j = phi double [ %chk.i, %outer.header ], [ %chk.next.j, %inner.header ]
  %aidx = getelementptr inbounds [1335 x double], ptr %A, i64 %j, i64 %i
  %av = load double, ptr %aidx, align 8
  %chk.next.j = fadd reassoc double %chk.j, %av
  %j.next = add i64 %j, 1
  %j.ec = icmp eq i64 %j.next, 1335
  br i1 %j.ec, label %epilogue.first, label %inner.header

epilogue.first:
  %chk.next = phi double [ %chk.next.j, %inner.header ]
    #dbg_value(double %chk.next, !7, !DIExpression(), !12)
    #dbg_value(double %en, !8, !DIExpression(), !13)
    #dbg_value(!DIArgList(double %en, double %outer.variant), !16, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), !18)
    #dbg_value(double 1.000000e+00, !17, !DIExpression(), !19)
  %dp = getelementptr inbounds double, ptr %D, i64 %i
  %dv = load double, ptr %dp, align 8
  %dn = fadd double %dv, 1.000000e+00
  store double %dn, ptr %dp, align 8
    #dbg_value(double %dn, !9, !DIExpression(), !14)
  br label %epilogue.second

epilogue.second:
  %ep = getelementptr inbounds double, ptr %EArray, i64 %i
  %ev = load double, ptr %ep, align 8
  %en = fadd double %ev, 2.000000e+00
  store double %en, ptr %ep, align 8
    #dbg_value(double %en, !10, !DIExpression(), !15)
  br label %outer.latch

outer.latch:
  %i.next = add i64 %i, 1
  %i.ec = icmp eq i64 %i.next, 1335
  br i1 %i.ec, label %exit, label %outer.header

exit:
  %chk.res = phi double [ %chk.next, %outer.latch ]
  store double %chk.res, ptr %R, align 8
  ret void
}

; The N forwarding-PHI location remains in the emptied connector. The three
; E-side records are absent here, so no erased E value can poison this path.
; CHECK-LABEL: define void @multi_block_debug(
; CHECK:       epilogue.first:
; CHECK-NEXT:      #dbg_value(double %chk.next, [[N_VAR:![0-9]+]],
; CHECK-NEXT:      #dbg_value(double 1.000000e+00, [[CONSTANT_VAR:![0-9]+]],
; CHECK-NEXT:    br label %outer.latch
;
; The use-before-def record is preserved before the first cloned instruction
; and remapped to the later %en.epil definition. The first path-terminator
; record lands at the next flattened block boundary, before %ep.epil. The final
; path-terminator record lands immediately before the new E-body branch.
; CHECK:       epilogue.header:
; CHECK:         %epilogue.iv = phi i64
; CHECK-NOT:     #dbg_value(!DIArgList(
; CHECK-NOT:     #dbg_value(double 1.000000e+00,
; CHECK-NEXT:      #dbg_value(double %en.epil, [[FUTURE_E_VAR:![0-9]+]],
; CHECK-NEXT:    %dp.epil = getelementptr inbounds double, ptr %D, i64 %epilogue.iv
; CHECK:         store double %dn.epil, ptr %dp.epil, align 8
; CHECK-NEXT:      #dbg_value(double %dn.epil, [[FIRST_TERM_VAR:![0-9]+]],
; CHECK-NEXT:    %ep.epil = getelementptr inbounds double, ptr %EArray, i64 %epilogue.iv
; CHECK:         store double %en.epil, ptr %ep.epil, align 8
; CHECK-NEXT:      #dbg_value(double %en.epil, [[FINAL_TERM_VAR:![0-9]+]],
; CHECK-NEXT:    br label %epilogue.latch
; CHECK-NOT:     #dbg_value(double %chk.next,

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "loop-interchange test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "outer-epilogue-fission-debug.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "multi_block_debug", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = !{!7, !8, !9, !10, !16, !17}
!7 = !DILocalVariable(name: "n_forward", scope: !3, file: !1, line: 2, type: !11)
!8 = !DILocalVariable(name: "future_e", scope: !3, file: !1, line: 3, type: !11)
!9 = !DILocalVariable(name: "first_term_e", scope: !3, file: !1, line: 4, type: !11)
!10 = !DILocalVariable(name: "final_term_e", scope: !3, file: !1, line: 5, type: !11)
!11 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!12 = !DILocation(line: 2, column: 1, scope: !3)
!13 = !DILocation(line: 3, column: 1, scope: !3)
!14 = !DILocation(line: 4, column: 1, scope: !3)
!15 = !DILocation(line: 5, column: 1, scope: !3)
!16 = !DILocalVariable(name: "mixed_outer_variant", scope: !3, file: !1, line: 6, type: !11)
!17 = !DILocalVariable(name: "constant_only", scope: !3, file: !1, line: 7, type: !11)
!18 = !DILocation(line: 6, column: 1, scope: !3)
!19 = !DILocation(line: 7, column: 1, scope: !3)
