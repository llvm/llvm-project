; RUN: opt < %s -S -passes='loop-vectorize,loop-unroll' \
; RUN:   -pass-remarks=loop-unroll \
; RUN:   -force-vector-width=4 -force-vector-interleave=1 \
; RUN:   -enable-epilogue-vectorization -epilogue-vectorization-force-VF=4 \
; RUN:   -unroll-count=2 -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -S -passes='loop-vectorize,loop-unroll' \
; RUN:   -pass-remarks=loop-unroll \
; RUN:   -force-vector-width=4 -force-vector-interleave=1 \
; RUN:   -enable-epilogue-vectorization -epilogue-vectorization-force-VF=4 \
; RUN:   -unroll-count=2 -disable-output -pass-remarks-output=%t.yaml 2>&1
; RUN: FileCheck --input-file=%t.yaml %s --check-prefix=YAML

; Verify that when the loop vectorizer produces body / epilogue
; metadata and the unroller successfully unrolls the resulting loops, each
; unroll remark carries the correct loop-kind qualifier.
;
; Pipeline: loop-vectorize -> loop-unroll (with forced unroll count).
; Epilogue vectorization is forced to exercise all four loop categories:
;
;   1. plain loop         – not touched by the vectorizer
;   2. vectorized loop    – main vector body      (has vectorize.body)
;   3. epilogue loop      – epilogue loop          (has vectorize.epilogue)
;
; Both stderr remarks and YAML structured output are checked.

;--- plain_loop: vectorize.enable=false keeps this a plain scalar loop --------

; CHECK:     remark: test.cpp:1:1: unrolled loop by a factor of 2
; CHECK-NOT: remark: test.cpp:1:1: unrolled vectorized
; CHECK-NOT: remark: test.cpp:1:1: unrolled epilogue

define void @plain_loop(ptr noalias %A, i64 %n) !dbg !100 {
entry:
  br label %loop, !dbg !108
loop:
  %i = phi i64 [0, %entry], [%i.next, %loop]
  %gep = getelementptr inbounds i32, ptr %A, i64 %i
  %v = load i32, ptr %gep, align 4
  %add = add i32 %v, 1
  store i32 %add, ptr %gep, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp slt i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit, !dbg !108, !llvm.loop !10
exit:
  ret void
}

;--- vectorizable_loop: vectorized with epilogue -> 3 unrolled sub-loops ------

; CHECK-DAG: remark: test.cpp:10:1: unrolled vectorized loop by a factor of 2
; CHECK-DAG: remark: test.cpp:10:1: unrolled epilogue loop by a factor of 2

define void @vectorizable_loop(ptr noalias %A, ptr noalias %B, i64 %n) !dbg !200 {
entry:
  br label %loop, !dbg !208
loop:
  %i = phi i64 [0, %entry], [%i.next, %loop]
  %gep.a = getelementptr inbounds i32, ptr %A, i64 %i
  %gep.b = getelementptr inbounds i32, ptr %B, i64 %i
  %v = load i32, ptr %gep.a, align 4
  %add = add i32 %v, 1
  store i32 %add, ptr %gep.b, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp slt i64 %i.next, %n
  br i1 %cmp, label %loop, label %exit, !dbg !208
exit:
  ret void
}

;--- YAML checks ---------------------------------------------------------------

; YAML:      --- !Passed
; YAML:      Pass:            loop-unroll
; YAML:      Name:            PartialUnrolled
; YAML:      Function:        plain_loop
; YAML:      Args:
; YAML:        - String:          'unrolled loop by a factor of '

; YAML:      --- !Passed
; YAML:      Pass:            loop-unroll
; YAML:      Name:            PartialUnrolled
; YAML:      Function:        vectorizable_loop
; YAML:      Args:
; YAML:        - String:          'unrolled vectorized loop by a factor of '

; YAML:      --- !Passed
; YAML:      Pass:            loop-unroll
; YAML:      Name:            PartialUnrolled
; YAML:      Function:        vectorizable_loop
; YAML:      Args:
; YAML:        - String:          'unrolled epilogue loop by a factor of '

;--- Metadata ------------------------------------------------------------------

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "test.cpp", directory: ".")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!99 = !DISubroutineType(types: !{})

!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.vectorize.enable", i1 false}

!100 = distinct !DISubprogram(name: "plain_loop", scope: !1, file: !1, line: 1, type: !99, isLocal: false, isDefinition: true, scopeLine: 1, unit: !0)
!108 = !DILocation(line: 1, column: 1, scope: !100)

!200 = distinct !DISubprogram(name: "vectorizable_loop", scope: !1, file: !1, line: 10, type: !99, isLocal: false, isDefinition: true, scopeLine: 10, unit: !0)
!208 = !DILocation(line: 10, column: 1, scope: !200)
