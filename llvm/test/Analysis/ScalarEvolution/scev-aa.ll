; RUN: opt -disable-output < %s -aa-pipeline=scev-aa -passes=aa-eval -print-all-alias-modref-info \
; RUN:   2>&1 | FileCheck %s

; At the time of this writing, misses the example of the form
; A[i+(j+1)] != A[i+j], which can arise from multi-dimensional array references,
; and the example of the form A[0] != A[i+1], where i+1 is known to be positive.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"

; p[i] and p[i+1] don't alias.

; CHECK-LABEL: Function: loop
; CHECK: NoAlias: double* %pi, double* %pi.next

define void @loop(ptr nocapture %p, i64 %n) nounwind {
entry:
  %j = icmp sgt i64 %n, 0
  br i1 %j, label %bb, label %return

bb:
  %i = phi i64 [ 0, %entry ], [ %i.next, %bb ]
  %pi = getelementptr double, ptr %p, i64 %i
  %i.next = add i64 %i, 1
  %pi.next = getelementptr double, ptr %p, i64 %i.next
  %x = load double, ptr %pi
  %y = load double, ptr %pi.next
  %z = fmul double %x, %y
  store double %z, ptr %pi
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %return, label %bb

return:
  ret void
}

; Slightly more involved: p[j][i], p[j][i+1], and p[j+1][i] don't alias.

; CHECK-LABEL: Function: nestedloop
; CHECK: NoAlias: double* %pi.j, double* %pi.next.j
; CHECK: NoAlias: double* %pi.j, double* %pi.j.next
; CHECK: NoAlias: double* %pi.j.next, double* %pi.next.j

define void @nestedloop(ptr nocapture %p, i64 %m) nounwind {
entry:
  %k = icmp sgt i64 %m, 0
  br i1 %k, label %guard, label %return

guard:
  %l = icmp sgt i64 91, 0
  br i1 %l, label %outer.loop, label %return

outer.loop:
  %j = phi i64 [ 0, %guard ], [ %j.next, %outer.latch ]
  br label %bb

bb:
  %i = phi i64 [ 0, %outer.loop ], [ %i.next, %bb ]
  %i.next = add i64 %i, 1

  %e = add i64 %i, %j
  %pi.j = getelementptr double, ptr %p, i64 %e
  %f = add i64 %i.next, %j
  %pi.next.j = getelementptr double, ptr %p, i64 %f
  %x = load double, ptr %pi.j
  %y = load double, ptr %pi.next.j
  %z = fmul double %x, %y
  store double %z, ptr %pi.j

  %o = add i64 %j, 91
  %g = add i64 %i, %o
  %pi.j.next = getelementptr double, ptr %p, i64 %g
  %a = load double, ptr %pi.j.next
  %b = fmul double %x, %a
  store double %b, ptr %pi.j.next

  %exitcond = icmp eq i64 %i.next, 91
  br i1 %exitcond, label %outer.latch, label %bb

outer.latch:
  %j.next = add i64 %j, 91
  %h = icmp eq i64 %j.next, %m
  br i1 %h, label %return, label %outer.loop

return:
  ret void
}

; Even more involved: same as nestedloop, but with a variable extent.
; When n is 1, p[j+1][i] does alias p[j][i+1], and there's no way to
; prove whether n will be greater than 1, so that relation will always
; by MayAlias. The loop is guarded by a n > 0 test though, so
; p[j+1][i] and p[j][i] can theoretically be determined to be NoAlias,
; however the analysis currently doesn't do that.
; TODO: Make the analysis smarter and turn that MayAlias into a NoAlias.

; CHECK-LABEL: Function: nestedloop_more
; CHECK: NoAlias: double* %pi.j, double* %pi.next.j
; CHECK: MayAlias: double* %pi.j, double* %pi.j.next

define void @nestedloop_more(ptr nocapture %p, i64 %n, i64 %m) nounwind {
entry:
  %k = icmp sgt i64 %m, 0
  br i1 %k, label %guard, label %return

guard:
  %l = icmp sgt i64 %n, 0
  br i1 %l, label %outer.loop, label %return

outer.loop:
  %j = phi i64 [ 0, %guard ], [ %j.next, %outer.latch ]
  br label %bb

bb:
  %i = phi i64 [ 0, %outer.loop ], [ %i.next, %bb ]
  %i.next = add i64 %i, 1

  %e = add i64 %i, %j
  %pi.j = getelementptr double, ptr %p, i64 %e
  %f = add i64 %i.next, %j
  %pi.next.j = getelementptr double, ptr %p, i64 %f
  %x = load double, ptr %pi.j
  %y = load double, ptr %pi.next.j
  %z = fmul double %x, %y
  store double %z, ptr %pi.j

  %o = add i64 %j, %n
  %g = add i64 %i, %o
  %pi.j.next = getelementptr double, ptr %p, i64 %g
  %a = load double, ptr %pi.j.next
  %b = fmul double %x, %a
  store double %b, ptr %pi.j.next

  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %outer.latch, label %bb

outer.latch:
  %j.next = add i64 %j, %n
  %h = icmp eq i64 %j.next, %m
  br i1 %h, label %return, label %outer.loop

return:
  ret void
}

; ScalarEvolution expands field offsets into constants, which allows it to
; do aggressive analysis. Contrast this with BasicAA, which works by
; recognizing GEP idioms.

%struct.A = type { %struct.B, i32, i32 }
%struct.B = type { double }

; CHECK-LABEL: Function: foo
; CHECK-DAG: NoAlias: %struct.B* %A, i32* %Z
; CHECK-DAG: NoAlias: %struct.B* %A, %struct.B* %C
; CHECK-DAG: MustAlias: %struct.B* %C, i32* %Z
; CHECK-DAG: NoAlias: %struct.B* %A, i32* %C
; CHECK-DAG: MustAlias: i32* %C, i32* %Z
; CHECK-DAG: MustAlias: %struct.B* %C, i32* %Y
; CHECK-DAG: MustAlias: i32* %C, i32* %Y

define void @foo() {
entry:
  %A = alloca %struct.A
  %Z = getelementptr %struct.A, ptr %A, i32 0, i32 1
  %C = getelementptr %struct.B, ptr %A, i32 1
  %Y = getelementptr %struct.A, ptr %A, i32 0, i32 1
  load %struct.B, ptr %A
  load %struct.B, ptr %C
  load i32, ptr %C
  load i32, ptr %Y
  load i32, ptr %Z
  ret void
}

; CHECK-LABEL: Function: bar
; CHECK-DAG: NoAlias: %struct.B* %M, i32* %P
; CHECK-DAG: NoAlias: %struct.B* %M, %struct.B* %R
; CHECK-DAG: MustAlias: i32* %P, %struct.B* %R
; CHECK-DAG: NoAlias: %struct.B* %M, i32* %R
; CHECK-DAG: MustAlias: i32* %P, i32* %R
; CHECK-DAG: MustAlias: %struct.B* %R, i32* %V
; CHECK-DAG: MustAlias: i32* %R, i32* %V

define void @bar() {
  %M = alloca %struct.A
  %P = getelementptr %struct.A, ptr %M, i32 0, i32 1
  %R = getelementptr %struct.B, ptr %M, i32 1
  %V = getelementptr %struct.A, ptr %M, i32 0, i32 1
  load %struct.B, ptr %M
  load %struct.B, ptr %R
  load i32, ptr %P
  load i32, ptr %V
  load i32, ptr %R
  ret void
}

; CHECK: Function: nonnegative: 2 pointers, 0 call sites
; CHECK: NoAlias:  i64* %arrayidx, i64* %p

define void @nonnegative(ptr %p) nounwind {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i64 [ %inc, %for.body ], [ 0, %entry ] ; <i64> [#uses=2]
  %inc = add nsw i64 %i, 1                         ; <i64> [#uses=2]
  %arrayidx = getelementptr inbounds i64, ptr %p, i64 %inc
  store i64 0, ptr %arrayidx
  %tmp6 = load i64, ptr %p                            ; <i64> [#uses=1]
  %cmp = icmp slt i64 %inc, %tmp6                 ; <i1> [#uses=1]
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; CHECK-LABEL: Function: test_no_dom: 3 pointers, 0 call sites
; CHECK: MayAlias:	double* %addr1, double* %data
; CHECK: NoAlias:	double* %addr2, double* %data
; CHECK: MayAlias:	double* %addr1, double* %addr2

; In this case, checking %addr1 and %add2 involves two addrecs in two
; different loops where neither dominates the other.  This used to crash
; because we expected the arguments to an AddExpr to have a strict
; dominance order.
define void @test_no_dom(ptr %data, i1 %arg) {
entry:
  load double, ptr %data
  br label %for.body
  
for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.latch ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br i1 %arg, label %subloop1, label %subloop2

subloop1:
  %iv1 = phi i32 [0, %for.body], [%iv1.next, %subloop1]
  %iv1.next = add i32 %iv1, 1
  %addr1 = getelementptr double, ptr %data, i32 %iv1
  store double 0.0, ptr %addr1
  %cmp1 = icmp slt i32 %iv1, 200
  br i1 %cmp1, label %subloop1, label %for.latch

subloop2:
  %iv2 = phi i32 [400, %for.body], [%iv2.next, %subloop2]
  %iv2.next = add i32 %iv2, 1
  %addr2 = getelementptr double, ptr %data, i32 %iv2
  store double 0.0, ptr %addr2
  %cmp2 = icmp slt i32 %iv2, 600
  br i1 %cmp2, label %subloop2, label %for.latch

for.latch:
  br label %for.body

for.end:
  ret void
}

declare ptr @get_addr(i32 %i)

; CHECK-LABEL: Function: test_no_dom2: 3 pointers, 2 call sites
; CHECK: MayAlias:	double* %addr1, double* %data
; CHECK: MayAlias:	double* %addr2, double* %data
; CHECK: MayAlias:	double* %addr1, double* %addr2

; In this case, checking %addr1 and %add2 involves two addrecs in two
; different loops where neither dominates the other.  This is analogous
; to test_no_dom, but involves SCEVUnknown as opposed to SCEVAddRecExpr.
define void @test_no_dom2(ptr %data, i1 %arg) {
entry:
  load double, ptr %data
  br label %for.body
  
for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.latch ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br i1 %arg, label %subloop1, label %subloop2

subloop1:
  %iv1 = phi i32 [0, %for.body], [%iv1.next, %subloop1]
  %iv1.next = add i32 %iv1, 1
  %addr1 = call ptr @get_addr(i32 %iv1)
  store double 0.0, ptr %addr1
  %cmp1 = icmp slt i32 %iv1, 200
  br i1 %cmp1, label %subloop1, label %for.latch

subloop2:
  %iv2 = phi i32 [400, %for.body], [%iv2.next, %subloop2]
  %iv2.next = add i32 %iv2, 1
  %addr2 = call ptr @get_addr(i32 %iv2)
  store double 0.0, ptr %addr2
  %cmp2 = icmp slt i32 %iv2, 600
  br i1 %cmp2, label %subloop2, label %for.latch

for.latch:
  br label %for.body

for.end:
  ret void
}


; CHECK-LABEL: Function: test_dom: 3 pointers, 0 call sites
; CHECK: MayAlias:	double* %addr1, double* %data
; CHECK: NoAlias:	double* %addr2, double* %data
; CHECK: NoAlias:	double* %addr1, double* %addr2

; This is a variant of test_non_dom where the second subloop is
; dominated by the first.  As a result of that, we can nest the
; addrecs and cancel out the %data base pointer.
define void @test_dom(ptr %data) {
entry:
  load double, ptr %data
  br label %for.body
  
for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.latch ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %subloop1

subloop1:
  %iv1 = phi i32 [0, %for.body], [%iv1.next, %subloop1]
  %iv1.next = add i32 %iv1, 1
  %addr1 = getelementptr double, ptr %data, i32 %iv1
  store double 0.0, ptr %addr1
  %cmp1 = icmp slt i32 %iv1, 200
  br i1 %cmp1, label %subloop1, label %subloop2

subloop2:
  %iv2 = phi i32 [400, %subloop1], [%iv2.next, %subloop2]
  %iv2.next = add i32 %iv2, 1
  %addr2 = getelementptr double, ptr %data, i32 %iv2
  store double 0.0, ptr %addr2
  %cmp2 = icmp slt i32 %iv2, 600
  br i1 %cmp2, label %subloop2, label %for.latch

for.latch:
  br label %for.body

for.end:
  ret void
}

; CHECK-LABEL: Function: test_different_pointer_bases_of_inttoptr: 2 pointers, 0 call sites
; CHECK:   NoAlias:	<16 x i8>* %tmp5, <16 x i8>* %tmp7

define void @test_different_pointer_bases_of_inttoptr() {
entry:
  br label %for.body

for.body:
  %tmp = phi i32 [ %next, %for.body ], [ 1, %entry ]
  %tmp1 = shl nsw i32 %tmp, 1
  %tmp2 = add nuw nsw i32 %tmp1, %tmp1
  %tmp3 = mul nsw i32 %tmp2, 1408
  %tmp4 = add nsw i32 %tmp3, 1408
  %tmp5 = getelementptr inbounds i8, ptr inttoptr (i32 1024 to ptr), i32 %tmp1
  %tmp6 = load <16 x i8>, ptr %tmp5, align 1
  %tmp7 = getelementptr inbounds i8, ptr inttoptr (i32 4096 to ptr), i32 %tmp4
  store <16 x i8> %tmp6, ptr %tmp7, align 1

  %next = add i32 %tmp, 2
  %exitcond = icmp slt i32 %next, 10
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}
