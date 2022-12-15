; RUN: opt -S -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 2>&1 | FileCheck %s

; umin is represented using -1 * umax in scalar evolution. -1 is considered as the
; constant of the multiply expression (-1 * ((-1 + (-1 * %a)) umax (-1 + (-1 * %b)))).
; Returns the greatest power of 2 divisor by evaluating the minimal trailing zeros
; for the trip count expression.
;
; int foo(uint32_t a, uint32_t b, uint32_t *c) {
;   for (uint32_t i = 0; i < (uint32_t)(a < b ? a : b) + 1; i++)
;     c[i] = i;
;   return 0;
; }
;
; CHECK: Loop %for.body: Trip multiple is 1

define i32 @foo(i32 %a, i32 %b, ptr %c) {
entry:
  %cmp = icmp ult i32 %a, %b
  %cond = select i1 %cmp, i32 %a, i32 %b
  %add = add i32 %cond, 1
  %cmp18 = icmp eq i32 %add, 0
  br i1 %cmp18, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret i32 0

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.09 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %c, i32 %i.09
  store i32 %i.09, ptr %arrayidx, align 4
  %inc = add nuw i32 %i.09, 1
  %cmp1 = icmp ult i32 %inc, %add
  br i1 %cmp1, label %for.body, label %for.cond.cleanup.loopexit
}

; Overflow may happen for the multiply expression n * 3, verify that trip
; multiple is set to 1 if NUW/NSW are not set.
;
; __attribute__((noinline)) void a(unsigned n) {
;   #pragma unroll(3)
;   for (unsigned i = 0; i != n * 3; ++i)
;     printf("TEST%u\n", i);
; }
; int main() { a(2863311531U); }
;
; CHECK: Loop %for.body: Trip multiple is 1

@.str2 = private unnamed_addr constant [8 x i8] c"TEST%u\0A\00", align 1

define void @foo2(i32 %n) {
entry:
  %mul = mul i32 %n, 3
  %cmp4 = icmp eq i32 %mul, 0
  br i1 %cmp4, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.05 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %call = tail call i32 (ptr, ...) @printf(ptr @.str2, i32 %i.05)
  %inc = add nuw i32 %i.05, 1
  %cmp = icmp eq i32 %inc, %mul
  br i1 %cmp, label %for.cond.cleanup.loopexit, label %for.body
}

declare i32 @printf(ptr nocapture readonly, ...)


; If we couldn't prove no overflow for the multiply expression 24 * n,
; returns the greatest power of 2 divisor. If overflows happens
; the trip count is still divisible by the greatest power of 2 divisor.
;
; CHECK: Loop %l3: Trip multiple is 8

declare void @f()

define i32 @foo3(i32 %n) {
entry:
  %loop_ctl = mul i32 %n, 24
  br label %l3

l3:
  %x.0 = phi i32 [ 0, %entry ], [ %inc, %l3 ]
  call void @f()
  %inc = add i32 %x.0, 1
  %exitcond = icmp eq i32 %inc, %loop_ctl
  br i1 %exitcond, label %exit, label %l3

exit:
  ret i32 0
}

; If the trip count is a constant, verify that we obtained the trip
; count itself. For huge trip counts, or zero, we return 1.
;
; CHECK: Loop %l3: Trip multiple is 3

define i32 @foo4(i32 %n) {
entry:
  br label %l3

l3:
  %x.0 = phi i32 [ 0, %entry ], [ %inc, %l3 ]
  call void @f()
  %inc = add i32 %x.0, 1
  %exitcond = icmp eq i32 %inc, 3
  br i1 %exitcond, label %exit, label %l3

exit:
  ret i32 0
}

; If there are multiple exits, the result is the GCD of the multiples
; of each individual exit (since we don't know which is taken).

; CHECK: Loop %l4: Trip multiple is 50

define i32 @foo5(i32 %n) {
entry:
  br label %l4

l4:
  %x.0 = phi i32 [ 0, %entry ], [ %inc, %l4-latch ]
  call void @f()
  %inc = add i32 %x.0, 1
  %earlycond = icmp eq i32 %inc, 150
  br i1 %earlycond, label %exit, label %l4-latch

l4-latch:
  %exitcond = icmp eq i32 %inc, 200
  br i1 %exitcond, label %exit, label %l4

exit:
  ret i32 0
}

