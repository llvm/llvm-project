; REQUIRES: asserts
; RUN: opt -passes=loop-distribute -enable-loop-distribute \
; RUN:   -debug-only=loop-distribute -disable-output 2>&1 %s | FileCheck %s

define void @f(ptr noalias %a, ptr noalias %b, ptr noalias %c, ptr noalias %d, i64 %stride) {
; CHECK-LABEL: 'f'
; CHECK:        LDist: Found a candidate loop: for.body
; CHECK:        Backward dependences:
; CHECK-NEXT:     Backward:
; CHECK-NEXT:         %load.a = load i32, ptr %gep.a, align 4 ->
; CHECK-NEXT:         store i32 %mul.a, ptr %gep.a.plus4, align 4
; CHECK:        Seeded partitions:
; CHECK:        Partition 0
; CHECK:        Partition 1
; CHECK:        Partition 2
; CHECK:        Partition 3
; CHECK:        Distributing loop
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %ind
  %load.a = load i32, ptr %gep.a, align 4
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %ind
  %load.b = load i32, ptr %gep.b, align 4
  %mul.a = mul i32 %load.b, %load.a
  %add = add nuw nsw i64 %ind, 1
  %gep.a.plus4 = getelementptr inbounds i32, ptr %a, i64 %add
  store i32 %mul.a, ptr %gep.a.plus4, align 4
  %gep.d = getelementptr inbounds i32, ptr %d, i64 %ind
  %loadD = load i32, ptr %gep.d, align 4
  %mul = mul i64 %ind, %stride
  %gep.strided.a = getelementptr inbounds i32, ptr %a, i64 %mul
  %load.strided.a = load i32, ptr %gep.strided.a, align 4
  %mul.c = mul i32 %loadD, %load.strided.a
  %gep.c = getelementptr inbounds i32, ptr %c, i64 %ind
  store i32 %mul.c, ptr %gep.c, align 4
  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret void
}
