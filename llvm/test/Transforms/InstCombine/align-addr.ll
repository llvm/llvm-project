; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {align 16} | count 1

; Instcombine should be able to prove vector alignment in the
; presence of a few mild address computation tricks.

define void @foo(i8* %b, i64 %n, i64 %u, i64 %y) nounwind  {
entry:
  %c = ptrtoint i8* %b to i64
  %d = and i64 %c, -16
  %e = inttoptr i64 %d to double*
  %v = mul i64 %u, 2
  %z = and i64 %y, -2
  %t1421 = icmp eq i64 %n, 0
  br i1 %t1421, label %return, label %bb

bb:
  %i = phi i64 [ %indvar.next, %bb ], [ 20, %entry ]
  %j = mul i64 %i, %v
  %h = add i64 %j, %z
  %t8 = getelementptr double* %e, i64 %h
  %p = bitcast double* %t8 to <2 x double>*
  store <2 x double><double 0.0, double 0.0>, <2 x double>* %p, align 8
  %indvar.next = add i64 %i, 1
  %exitcond = icmp eq i64 %indvar.next, %n
  br i1 %exitcond, label %return, label %bb

return:
  ret void
}

