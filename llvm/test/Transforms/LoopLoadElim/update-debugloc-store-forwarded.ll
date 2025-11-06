; RUN: opt -passes=loop-load-elim -S < %s | FileCheck %s

; LoopLoadElimination's propagateStoredValueToLoadUsers() replaces the
; `load` (`%a`) with a hoisted initial `load` and a `phi` that forwards
; the stored value.
; This test checks that the debug location is propagated to the new `phi`
; from the original `load` it replaces in block `%for.body` and the debug
; location drop of the hoisted `load` in block `%entry`.
; Moreover, this test also checks the debug location update of the new
; `bitcast` created when the `load` type is mismatched with the `store` type:
;   store i32 ...
;   %a = load float, ...
; Because the `bitcast` casts the old `load` value, it has the same debug
; location as the old `load` (ie., the same as the new `phi`).

; If the store and the load use different types, but have the same
; size then we should still be able to forward the value.
;
;   for (unsigned i = 0; i < 100; i++) {
;     A[i+1] = B[i] + 2;
;     C[i] = ((float*)A)[i] * 2;
;   }

define void @f(ptr noalias %A, ptr noalias %B, ptr noalias %C, i64 %N) !dbg !5 {
; CHECK-LABEL: define void @f(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[LOAD_INITIAL:%.*]] = load float, ptr {{.*}}, align 4{{$}}
; CHECK:       for.body:
; CHECK-NEXT:    [[STORE_FORWARDED:%.*]] = phi float [ [[LOAD_INITIAL]], %entry ], [ [[STORE_FORWARD_CAST:%.*]], %for.body ], !dbg [[DBG9:![0-9]+]]
; CHECK:         [[STORE_FORWARD_CAST]] = bitcast i32 {{.*}} to float, !dbg [[DBG9]]
; CHECK:       [[DBG9]] = !DILocation(line: 11,
;
entry:
  br label %for.body, !dbg !8

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ], !dbg !9
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !10
  %Aidx_next = getelementptr inbounds i32, ptr %A, i64 %indvars.iv.next, !dbg !11
  %Bidx = getelementptr inbounds i32, ptr %B, i64 %indvars.iv, !dbg !12
  %Cidx = getelementptr inbounds i32, ptr %C, i64 %indvars.iv, !dbg !13
  %Aidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv, !dbg !14
  %b = load i32, ptr %Bidx, align 4, !dbg !15
  %a_p1 = add i32 %b, 2, !dbg !16
  store i32 %a_p1, ptr %Aidx_next, align 4, !dbg !17
  %a = load float, ptr %Aidx, align 4, !dbg !18
  %c = fmul float %a, 2.000000e+00, !dbg !19
  %c.int = fptosi float %c to i32, !dbg !20
  store i32 %c.int, ptr %Cidx, align 4, !dbg !21
  %exitcond = icmp eq i64 %indvars.iv.next, %N, !dbg !22
  br i1 %exitcond, label %for.end, label %for.body, !dbg !23

for.end:                                          ; preds = %for.body
  ret void, !dbg !24
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "type-mismatch.ll", directory: "/")
!2 = !{i32 17}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5)
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DILocation(line: 3, column: 1, scope: !5)
!11 = !DILocation(line: 4, column: 1, scope: !5)
!12 = !DILocation(line: 5, column: 1, scope: !5)
!13 = !DILocation(line: 6, column: 1, scope: !5)
!14 = !DILocation(line: 7, column: 1, scope: !5)
!15 = !DILocation(line: 8, column: 1, scope: !5)
!16 = !DILocation(line: 9, column: 1, scope: !5)
!17 = !DILocation(line: 10, column: 1, scope: !5)
!18 = !DILocation(line: 11, column: 1, scope: !5)
!19 = !DILocation(line: 12, column: 1, scope: !5)
!20 = !DILocation(line: 13, column: 1, scope: !5)
!21 = !DILocation(line: 14, column: 1, scope: !5)
!22 = !DILocation(line: 15, column: 1, scope: !5)
!23 = !DILocation(line: 16, column: 1, scope: !5)
!24 = !DILocation(line: 17, column: 1, scope: !5)
