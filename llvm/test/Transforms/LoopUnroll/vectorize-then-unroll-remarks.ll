; RUN: opt < %s -S -passes='loop-vectorize,loop-unroll' \
; RUN:   -pass-remarks=loop-unroll -pass-remarks-missed=loop-unroll \
; RUN:   -force-vector-width=4 -unroll-count=2 2>&1 | FileCheck %s

; End-to-end test: the vectorizer creates a vectorized main loop and a scalar
; remainder.  When the unroller processes both, it should produce distinct
; remarks for each, using the metadata the vectorizer attached.

; The vectorized loop should get the "vectorized loop" qualifier.
; CHECK-DAG: remark: <stdin>:1:1: unrolled vectorized loop by a factor of 2
; The remainder should get the "scalar remainder loop after vectorization" qualifier.
; CHECK-DAG: remark: <stdin>:1:1: {{(unrolled scalar remainder loop after vectorization|scalar remainder loop after vectorization is not unrolled)}}

define void @vec_then_unroll(ptr noalias %a, ptr noalias %b, i64 %n) !dbg !6 {
entry:
  br label %for.body, !dbg !8

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep.a = getelementptr inbounds float, ptr %a, i64 %iv
  %gep.b = getelementptr inbounds float, ptr %b, i64 %iv
  %load = load float, ptr %gep.a, align 4
  %mul = fmul float %load, 2.0
  store float %mul, ptr %gep.b, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp = icmp slt i64 %iv.next, %n
  br i1 %cmp, label %for.body, label %for.end, !dbg !8

for.end:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "<stdin>", directory: ".")
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "vec_then_unroll", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, unit: !0)
!7 = !DISubroutineType(types: !{})
!8 = !DILocation(line: 1, column: 1, scope: !6)
