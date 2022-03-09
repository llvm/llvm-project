; RUN: opt < %s -aa-pipeline=basic-aa -passes='require<phi-values>,aa-eval' -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; rdar://7282591

@X = common global i32 0
@Y = common global i32 0
@Z = common global i32 0

; CHECK-LABEL: foo
; CHECK:  NoAlias: i32* %P, i32* @Z

define void @foo(i32 %cond) nounwind {
entry:
  %"alloca point" = bitcast i32 0 to i32
  %tmp = icmp ne i32 %cond, 0
  br i1 %tmp, label %bb, label %bb1

bb:
  br label %bb2

bb1:
  br label %bb2

bb2:
  %P = phi i32* [ @X, %bb ], [ @Y, %bb1 ]
  %tmp1 = load i32, i32* @Z, align 4
  store i32 123, i32* %P, align 4
  %tmp2 = load i32, i32* @Z, align 4
  br label %return

return:
  ret void
}

; Pointers can vary in between iterations of loops.
; PR18068

; CHECK-LABEL: pr18068
; CHECK: MayAlias: i32* %0, i32* %arrayidx5
; CHECK: NoAlias: i32* %arrayidx13, i32* %arrayidx5

define i32 @pr18068(i32* %jj7, i32* %j) {
entry:
  %oa5 = alloca [100 x i32], align 16
  br label %codeRepl

codeRepl:
  %0 = phi i32* [ %arrayidx13, %for.body ], [ %j, %entry ]
  %targetBlock = call i1 @cond(i32* %jj7)
  br i1 %targetBlock, label %for.body, label %bye

for.body:
  %1 = load i32, i32* %jj7, align 4
  %idxprom4 = zext i32 %1 to i64
  %arrayidx5 = getelementptr inbounds [100 x i32], [100 x i32]* %oa5, i64 0, i64 %idxprom4
  %2 = load i32, i32* %arrayidx5, align 4
  %sub6 = sub i32 %2, 6
  store i32 %sub6, i32* %arrayidx5, align 4
  ; %0 and %arrayidx5 can alias! It is not safe to DSE the above store.
  %3 = load i32, i32* %0, align 4
  store i32 %3, i32* %arrayidx5, align 4
  %sub11 = add i32 %1, -1
  %idxprom12 = zext i32 %sub11 to i64
  %arrayidx13 = getelementptr inbounds [100 x i32], [100 x i32]* %oa5, i64 0, i64 %idxprom12
  call void @inc(i32* %jj7)
  br label %codeRepl

bye:
  %.reload = load i32, i32* %jj7, align 4
  ret i32 %.reload
}

declare i1 @cond(i32*)

declare void @inc(i32*)


; When we have a chain of phis in nested loops we should recognise if there's
; actually only one underlying value.
; CHECK-LABEL: loop_phi_chain
; CHECK: NoAlias: i32* %val1, i32* @Y
; CHECK: NoAlias: i32* %val2, i32* @Y
; CHECK: NoAlias: i32* %val3, i32* @Y
define void @loop_phi_chain(i32 %a, i32 %b, i32 %c) {
entry:
  br label %loop1

loop1:
  %n1 = phi i32 [ 0, %entry ], [ %add1, %loop2 ]
  %val1 = phi i32* [ @X, %entry ], [ %val2, %loop2 ]
  %add1 = add i32 %n1, 1
  %cmp1 = icmp ne i32 %n1, 32
  br i1 %cmp1, label %loop2, label %end

loop2:
  %n2 = phi i32 [ 0, %loop1 ], [ %add2, %loop3 ]
  %val2 = phi i32* [ %val1, %loop1 ], [ %val3, %loop3 ]
  %add2 = add i32 %n2, 1
  %cmp2 = icmp ne i32 %n2, 32
  br i1 %cmp2, label %loop3, label %loop1

loop3:
  %n3 = phi i32 [ 0, %loop2 ], [ %add3, %loop3 ]
  %val3 = phi i32* [ %val2, %loop2 ], [ %val3, %loop3 ]
  store i32 0, i32* %val3, align 4
  store i32 0, i32* @Y, align 4
  %add3 = add i32 %n3, 1
  %cmp3 = icmp ne i32 %n3, 32
  br i1 %cmp3, label %loop3, label %loop2

end:
  ret void
}

; CHECK-LABEL: phi_and_select
; CHECK: MustAlias: i32* %p, i32* %s
define void @phi_and_select(i1 %c, i1 %c2, i32* %x, i32* %y) {
entry:
  br i1 %c, label %true, label %false

true:
  br label %exit

false:
  br label %exit

exit:
  %p = phi i32* [ %x, %true ], [ %y, %false ]
  %s = select i1 %c2, i32* %p, i32* %p
  store i32 0, i32* %p
  store i32 0, i32* %s
  ret void
}

; CHECK-LABEL: phi_and_phi_cycle
; CHECK: NoAlias: i32* %p1, i32* %p2
define void @phi_and_phi_cycle(i32* noalias %x, i32* noalias %y) {
entry:
  br label %loop

loop:
  %p1 = phi i32* [ %x, %entry ], [ %p1.next, %loop ]
  %p2 = phi i32* [ %y, %entry ], [ %p2.next, %loop ]
  %p1.next = getelementptr i32, i32* %p1, i64 1
  %p2.next = getelementptr i32, i32* %p1, i64 2
  store i32 0, i32* %p1
  store i32 0, i32* %p2
  br label %loop
}

; CHECK-LABEL: phi_and_gep_unknown_size
; CHECK: Just Mod:   call void @llvm.memset.p0i8.i32(i8* %g, i8 0, i32 %size, i1 false) <->   call void @llvm.memset.p0i8.i32(i8* %z, i8 0, i32 %size, i1 false)
; TODO: This should be NoModRef.
define void @phi_and_gep_unknown_size(i1 %c, i8* %x, i8* %y, i8* noalias %z, i32 %size) {
entry:
  br i1 %c, label %true, label %false

true:
  br label %exit

false:
  br label %exit

exit:
  %p = phi i8* [ %x, %true ], [ %y, %false ]
  %g = getelementptr inbounds i8, i8* %p, i64 1
  call void @llvm.memset.p0i8.i32(i8* %g, i8 0, i32 %size, i1 false)
  call void @llvm.memset.p0i8.i32(i8* %z, i8 0, i32 %size, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i32(i8*, i8, i32, i1)

; CHECK-LABEL: unsound_inequality
; CHECK: MayAlias:  i32* %arrayidx13, i32* %phi
; CHECK: MayAlias:  i32* %arrayidx5, i32* %phi
; CHECK: NoAlias:   i32* %arrayidx13, i32* %arrayidx5

; When recursively reasoning about phis, we can't use predicates between
; two values as we might be comparing the two from different iterations.
define i32 @unsound_inequality(i32* %jj7, i32* %j) {
entry:
  %oa5 = alloca [100 x i32], align 16
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %phi = phi i32* [ %arrayidx13, %for.body ], [ %j, %entry ]
  %idx = load i32, i32* %jj7, align 4
  %arrayidx5 = getelementptr inbounds [100 x i32], [100 x i32]* %oa5, i64 0, i32 %idx
  store i32 0, i32* %arrayidx5, align 4
  store i32 0, i32* %phi, align 4
  %notequal = add i32 %idx, 1
  %arrayidx13 = getelementptr inbounds [100 x i32], [100 x i32]* %oa5, i64 0, i32 %notequal
  store i32 0, i32* %arrayidx13, align 4
  br label %for.body
} 

; CHECK-LABEL: single_arg_phi
; CHECK: NoAlias: i32* %ptr, i32* %ptr.next
; CHECK: MustAlias: i32* %ptr, i32* %ptr.phi
; CHECK: MustAlias: i32* %ptr.next, i32* %ptr.next.phi
define void @single_arg_phi(i32* %ptr.base) {
entry:
  br label %loop

loop:
  %ptr = phi i32* [ %ptr.base, %entry ], [ %ptr.next, %split ]
  %ptr.next = getelementptr inbounds i32, i32* %ptr, i64 1
  br label %split

split:
  %ptr.phi = phi i32* [ %ptr, %loop ]
  %ptr.next.phi = phi i32* [ %ptr.next, %loop ]
  br label %loop
}
