; XFAIL: *
; ...should pass. See PR12324: misched bringup
; RUN: llc < %s -O3 -asm-verbose=false | FileCheck %s
target datalayout = "e-p:64:64:64"
target triple = "x86_64-unknown-unknown"

; Full strength reduction reduces register pressure from 5 to 4 here.
; Instruction selection should use the FLAGS value from the dec for
; the branch. Scheduling should push the adds upwards.

; CHECK-LABEL: full_me_0:
; CHECK: movsd   (%rsi), %xmm0
; CHECK: mulsd   (%rdx), %xmm0
; CHECK: movsd   %xmm0, (%rdi)
; CHECK: addq    $8, %rsi
; CHECK: addq    $8, %rdx
; CHECK: addq    $8, %rdi
; CHECK: decq    %rcx
; CHECK: jne

define void @full_me_0(ptr nocapture %A, ptr nocapture %B, ptr nocapture %C, i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %Ai = getelementptr inbounds double, ptr %A, i64 %i
  %Bi = getelementptr inbounds double, ptr %B, i64 %i
  %Ci = getelementptr inbounds double, ptr %C, i64 %i
  %t1 = load double, ptr %Bi
  %t2 = load double, ptr %Ci
  %m = fmul double %t1, %t2
  store double %m, ptr %Ai
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %return, label %loop

return:
  ret void
}

; Mostly-full strength reduction means we do full strength reduction on all
; except for the offsets.
;
; Given a choice between constant offsets -2048 and 2048, choose the negative
; value, because at boundary conditions it has a smaller encoding.
; TODO: That's an over-general heuristic. It would be better for the target
; to indicate what the encoding cost would be. Then using a 2048 offset
; would be better on x86-64, since the start value would be 0 instead of
; 2048.

; CHECK-LABEL: mostly_full_me_0:
; CHECK: movsd   -2048(%rsi), %xmm0
; CHECK: mulsd   -2048(%rdx), %xmm0
; CHECK: movsd   %xmm0, -2048(%rdi)
; CHECK: movsd   (%rsi), %xmm0
; CHECK: divsd   (%rdx), %xmm0
; CHECK: movsd   %xmm0, (%rdi)
; CHECK: addq    $8, %rsi
; CHECK: addq    $8, %rdx
; CHECK: addq    $8, %rdi
; CHECK: decq    %rcx
; CHECK: jne

define void @mostly_full_me_0(ptr nocapture %A, ptr nocapture %B, ptr nocapture %C, i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %Ai = getelementptr inbounds double, ptr %A, i64 %i
  %Bi = getelementptr inbounds double, ptr %B, i64 %i
  %Ci = getelementptr inbounds double, ptr %C, i64 %i
  %t1 = load double, ptr %Bi
  %t2 = load double, ptr %Ci
  %m = fmul double %t1, %t2
  store double %m, ptr %Ai
  %j = add i64 %i, 256
  %Aj = getelementptr inbounds double, ptr %A, i64 %j
  %Bj = getelementptr inbounds double, ptr %B, i64 %j
  %Cj = getelementptr inbounds double, ptr %C, i64 %j
  %t3 = load double, ptr %Bj
  %t4 = load double, ptr %Cj
  %o = fdiv double %t3, %t4
  store double %o, ptr %Aj
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %return, label %loop

return:
  ret void
}

; A minor variation on mostly_full_me_0.
; Prefer to start the indvar at 0.

; CHECK-LABEL: mostly_full_me_1:
; CHECK: movsd   (%rsi), %xmm0
; CHECK: mulsd   (%rdx), %xmm0
; CHECK: movsd   %xmm0, (%rdi)
; CHECK: movsd   -2048(%rsi), %xmm0
; CHECK: divsd   -2048(%rdx), %xmm0
; CHECK: movsd   %xmm0, -2048(%rdi)
; CHECK: addq    $8, %rsi
; CHECK: addq    $8, %rdx
; CHECK: addq    $8, %rdi
; CHECK: decq    %rcx
; CHECK: jne

define void @mostly_full_me_1(ptr nocapture %A, ptr nocapture %B, ptr nocapture %C, i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %Ai = getelementptr inbounds double, ptr %A, i64 %i
  %Bi = getelementptr inbounds double, ptr %B, i64 %i
  %Ci = getelementptr inbounds double, ptr %C, i64 %i
  %t1 = load double, ptr %Bi
  %t2 = load double, ptr %Ci
  %m = fmul double %t1, %t2
  store double %m, ptr %Ai
  %j = sub i64 %i, 256
  %Aj = getelementptr inbounds double, ptr %A, i64 %j
  %Bj = getelementptr inbounds double, ptr %B, i64 %j
  %Cj = getelementptr inbounds double, ptr %C, i64 %j
  %t3 = load double, ptr %Bj
  %t4 = load double, ptr %Cj
  %o = fdiv double %t3, %t4
  store double %o, ptr %Aj
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %return, label %loop

return:
  ret void
}

; A slightly less minor variation on mostly_full_me_0.

; CHECK-LABEL: mostly_full_me_2:
; CHECK: movsd   (%rsi), %xmm0
; CHECK: mulsd   (%rdx), %xmm0
; CHECK: movsd   %xmm0, (%rdi)
; CHECK: movsd   -4096(%rsi), %xmm0
; CHECK: divsd   -4096(%rdx), %xmm0
; CHECK: movsd   %xmm0, -4096(%rdi)
; CHECK: addq    $8, %rsi
; CHECK: addq    $8, %rdx
; CHECK: addq    $8, %rdi
; CHECK: decq    %rcx
; CHECK: jne

define void @mostly_full_me_2(ptr nocapture %A, ptr nocapture %B, ptr nocapture %C, i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %k = add i64 %i, 256
  %Ak = getelementptr inbounds double, ptr %A, i64 %k
  %Bk = getelementptr inbounds double, ptr %B, i64 %k
  %Ck = getelementptr inbounds double, ptr %C, i64 %k
  %t1 = load double, ptr %Bk
  %t2 = load double, ptr %Ck
  %m = fmul double %t1, %t2
  store double %m, ptr %Ak
  %j = sub i64 %i, 256
  %Aj = getelementptr inbounds double, ptr %A, i64 %j
  %Bj = getelementptr inbounds double, ptr %B, i64 %j
  %Cj = getelementptr inbounds double, ptr %C, i64 %j
  %t3 = load double, ptr %Bj
  %t4 = load double, ptr %Cj
  %o = fdiv double %t3, %t4
  store double %o, ptr %Aj
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %return, label %loop

return:
  ret void
}

; In this test, the counting IV exit value is used, so full strength reduction
; would not reduce register pressure. IndVarSimplify ought to simplify such
; cases away, but it's useful here to verify that LSR's register pressure
; heuristics are working as expected.

; CHECK-LABEL: count_me_0:
; CHECK: movsd   (%rsi,%rax,8), %xmm0
; CHECK: mulsd   (%rdx,%rax,8), %xmm0
; CHECK: movsd   %xmm0, (%rdi,%rax,8)
; CHECK: incq    %rax
; CHECK: cmpq    %rax, %rcx
; CHECK: jne

define i64 @count_me_0(ptr nocapture %A, ptr nocapture %B, ptr nocapture %C, i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %Ai = getelementptr inbounds double, ptr %A, i64 %i
  %Bi = getelementptr inbounds double, ptr %B, i64 %i
  %Ci = getelementptr inbounds double, ptr %C, i64 %i
  %t1 = load double, ptr %Bi
  %t2 = load double, ptr %Ci
  %m = fmul double %t1, %t2
  store double %m, ptr %Ai
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %return, label %loop

return:
  %q = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  ret i64 %q
}

; In this test, the trip count value is used, so full strength reduction
; would not reduce register pressure.
; (though it would reduce register pressure inside the loop...)

; CHECK-LABEL: count_me_1:
; CHECK: movsd   (%rsi,%rax,8), %xmm0
; CHECK: mulsd   (%rdx,%rax,8), %xmm0
; CHECK: movsd   %xmm0, (%rdi,%rax,8)
; CHECK: incq    %rax
; CHECK: cmpq    %rax, %rcx
; CHECK: jne

define i64 @count_me_1(ptr nocapture %A, ptr nocapture %B, ptr nocapture %C, i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  %Ai = getelementptr inbounds double, ptr %A, i64 %i
  %Bi = getelementptr inbounds double, ptr %B, i64 %i
  %Ci = getelementptr inbounds double, ptr %C, i64 %i
  %t1 = load double, ptr %Bi
  %t2 = load double, ptr %Ci
  %m = fmul double %t1, %t2
  store double %m, ptr %Ai
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %return, label %loop

return:
  %q = phi i64 [ 0, %entry ], [ %n, %loop ]
  ret i64 %q
}

; Full strength reduction doesn't save any registers here because the
; loop tripcount is a constant.

; CHECK-LABEL: count_me_2:
; CHECK: movl    $10, %eax
; CHECK: align
; CHECK: BB6_1:
; CHECK: movsd   -40(%rdi,%rax,8), %xmm0
; CHECK: addsd   -40(%rsi,%rax,8), %xmm0
; CHECK: movsd   %xmm0, -40(%rdx,%rax,8)
; CHECK: movsd   (%rdi,%rax,8), %xmm0
; CHECK: subsd   (%rsi,%rax,8), %xmm0
; CHECK: movsd   %xmm0, (%rdx,%rax,8)
; CHECK: incq    %rax
; CHECK: cmpq    $5010, %rax
; CHECK: jne

define void @count_me_2(ptr nocapture %A, ptr nocapture %B, ptr nocapture %C) nounwind {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %i5 = add i64 %i, 5
  %Ai = getelementptr double, ptr %A, i64 %i5
  %t2 = load double, ptr %Ai
  %Bi = getelementptr double, ptr %B, i64 %i5
  %t4 = load double, ptr %Bi
  %t5 = fadd double %t2, %t4
  %Ci = getelementptr double, ptr %C, i64 %i5
  store double %t5, ptr %Ci
  %i10 = add i64 %i, 10
  %Ai10 = getelementptr double, ptr %A, i64 %i10
  %t9 = load double, ptr %Ai10
  %Bi10 = getelementptr double, ptr %B, i64 %i10
  %t11 = load double, ptr %Bi10
  %t12 = fsub double %t9, %t11
  %Ci10 = getelementptr double, ptr %C, i64 %i10
  store double %t12, ptr %Ci10
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, 5000
  br i1 %exitcond, label %return, label %loop

return:
  ret void
}

; This should be fully strength-reduced to reduce register pressure.

; CHECK-LABEL: full_me_1:
; CHECK: align
; CHECK: BB7_1:
; CHECK: movsd   (%rdi), %xmm0
; CHECK: addsd   (%rsi), %xmm0
; CHECK: movsd   %xmm0, (%rdx)
; CHECK: movsd   40(%rdi), %xmm0
; CHECK: subsd   40(%rsi), %xmm0
; CHECK: movsd   %xmm0, 40(%rdx)
; CHECK: addq    $8, %rdi
; CHECK: addq    $8, %rsi
; CHECK: addq    $8, %rdx
; CHECK: decq    %rcx
; CHECK: jne

define void @full_me_1(ptr nocapture %A, ptr nocapture %B, ptr nocapture %C, i64 %n) nounwind {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %i5 = add i64 %i, 5
  %Ai = getelementptr double, ptr %A, i64 %i5
  %t2 = load double, ptr %Ai
  %Bi = getelementptr double, ptr %B, i64 %i5
  %t4 = load double, ptr %Bi
  %t5 = fadd double %t2, %t4
  %Ci = getelementptr double, ptr %C, i64 %i5
  store double %t5, ptr %Ci
  %i10 = add i64 %i, 10
  %Ai10 = getelementptr double, ptr %A, i64 %i10
  %t9 = load double, ptr %Ai10
  %Bi10 = getelementptr double, ptr %B, i64 %i10
  %t11 = load double, ptr %Bi10
  %t12 = fsub double %t9, %t11
  %Ci10 = getelementptr double, ptr %C, i64 %i10
  store double %t12, ptr %Ci10
  %i.next = add i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %return, label %loop

return:
  ret void
}

; This is a variation on full_me_0 in which the 0,+,1 induction variable
; has a non-address use, pinning that value in a register.

; CHECK-LABEL: count_me_3:
; CHECK: call
; CHECK: movsd   (%r{{[^,]*}},%r{{[^,]*}},8), %xmm0
; CHECK: mulsd   (%r{{[^,]*}},%r{{[^,]*}},8), %xmm0
; CHECK: movsd   %xmm0, (%r{{[^,]*}},%r{{[^,]*}},8)
; CHECK: incq    %r{{.*}}
; CHECK: cmpq    %r{{.*}}, %r{{.*}}
; CHECK: jne

declare void @use(i64)

define void @count_me_3(ptr nocapture %A, ptr nocapture %B, ptr nocapture %C, i64 %n) nounwind {
entry:
  %t0 = icmp sgt i64 %n, 0
  br i1 %t0, label %loop, label %return

loop:
  %i = phi i64 [ %i.next, %loop ], [ 0, %entry ]
  call void @use(i64 %i)
  %Ai = getelementptr inbounds double, ptr %A, i64 %i
  %Bi = getelementptr inbounds double, ptr %B, i64 %i
  %Ci = getelementptr inbounds double, ptr %C, i64 %i
  %t1 = load double, ptr %Bi
  %t2 = load double, ptr %Ci
  %m = fmul double %t1, %t2
  store double %m, ptr %Ai
  %i.next = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %n
  br i1 %exitcond, label %return, label %loop

return:
  ret void
}

; LSR should use only one indvar for the inner loop.
; rdar://7657764

; CHECK-LABEL: asd:
; CHECK: BB9_4:
; CHECK-NEXT: addl  (%r{{[^,]*}},%rdi,4), %e
; CHECK-NEXT: incq  %rdi
; CHECK-NEXT: cmpq  %rdi, %r{{[^,]*}}
; CHECK-NEXT: jg

%struct.anon = type { i32, [4200 x i32] }

@bars = common global [123123 x %struct.anon] zeroinitializer, align 32 ; <ptr> [#uses=2]

define i32 @asd(i32 %n) nounwind readonly {
entry:
  %0 = icmp sgt i32 %n, 0                         ; <i1> [#uses=1]
  br i1 %0, label %bb.nph14, label %bb5

bb.nph14:                                         ; preds = %entry
  %tmp18 = zext i32 %n to i64                     ; <i64> [#uses=1]
  br label %bb

bb:                                               ; preds = %bb3, %bb.nph14
  %indvar16 = phi i64 [ 0, %bb.nph14 ], [ %indvar.next17, %bb3 ] ; <i64> [#uses=3]
  %s.113 = phi i32 [ 0, %bb.nph14 ], [ %s.0.lcssa, %bb3 ] ; <i32> [#uses=2]
  %scevgep2526 = getelementptr [123123 x %struct.anon], ptr @bars, i64 0, i64 %indvar16, i32 0 ; <ptr> [#uses=1]
  %1 = load i32, ptr %scevgep2526, align 4            ; <i32> [#uses=2]
  %2 = icmp sgt i32 %1, 0                         ; <i1> [#uses=1]
  br i1 %2, label %bb.nph, label %bb3

bb.nph:                                           ; preds = %bb
  %tmp23 = sext i32 %1 to i64                     ; <i64> [#uses=1]
  br label %bb1

bb1:                                              ; preds = %bb.nph, %bb1
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp19, %bb1 ] ; <i64> [#uses=2]
  %s.07 = phi i32 [ %s.113, %bb.nph ], [ %4, %bb1 ] ; <i32> [#uses=1]
  %c.08 = getelementptr [123123 x %struct.anon], ptr @bars, i64 0, i64 %indvar16, i32 1, i64 %indvar ; <ptr> [#uses=1]
  %3 = load i32, ptr %c.08, align 4                   ; <i32> [#uses=1]
  %4 = add nsw i32 %3, %s.07                      ; <i32> [#uses=2]
  %tmp19 = add i64 %indvar, 1                     ; <i64> [#uses=2]
  %5 = icmp sgt i64 %tmp23, %tmp19                ; <i1> [#uses=1]
  br i1 %5, label %bb1, label %bb3

bb3:                                              ; preds = %bb1, %bb
  %s.0.lcssa = phi i32 [ %s.113, %bb ], [ %4, %bb1 ] ; <i32> [#uses=2]
  %indvar.next17 = add i64 %indvar16, 1           ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %indvar.next17, %tmp18  ; <i1> [#uses=1]
  br i1 %exitcond, label %bb5, label %bb

bb5:                                              ; preds = %bb3, %entry
  %s.1.lcssa = phi i32 [ 0, %entry ], [ %s.0.lcssa, %bb3 ] ; <i32> [#uses=1]
  ret i32 %s.1.lcssa
}

; Two loops here are of particular interest; the one at %bb21, where
; we don't want to leave extra induction variables around, or use an
; lea to compute an exit condition inside the loop:

; CHECK-LABEL: test:

; CHECK:      BB10_4:
; CHECK-NEXT:   movaps  %xmm{{.*}}, %xmm{{.*}}
; CHECK-NEXT:   addss   %xmm{{.*}}, %xmm{{.*}}
; CHECK-NEXT:   mulss   (%r{{[^,]*}}), %xmm{{.*}}
; CHECK-NEXT:   movss   %xmm{{.*}}, (%r{{[^,]*}})
; CHECK-NEXT:   addq    $4, %r{{.*}}
; CHECK-NEXT:   decq    %r{{.*}}
; CHECK-NEXT:   addq    $4, %r{{.*}}
; CHECK-NEXT:   movaps  %xmm{{.*}}, %xmm{{.*}}
; CHECK-NEXT: BB10_2:
; CHECK-NEXT:   testq   %r{{.*}}, %r{{.*}}
; CHECK-NEXT:   jle
; CHECK-NEXT:   testb   $15, %r{{.*}}
; CHECK-NEXT:   jne

; And the one at %bb68, where we want to be sure to use superhero mode:

; CHECK:      BB10_7:
; CHECK-NEXT:   movaps  48(%r{{[^,]*}}), %xmm{{.*}}
; CHECK-NEXT:   mulps   %xmm{{.*}}, %xmm{{.*}}
; CHECK-NEXT:   movaps  32(%r{{[^,]*}}), %xmm{{.*}}
; CHECK-NEXT:   mulps   %xmm{{.*}}, %xmm{{.*}}
; CHECK-NEXT:   movaps  16(%r{{[^,]*}}), %xmm{{.*}}
; CHECK-NEXT:   mulps   %xmm{{.*}}, %xmm{{.*}}
; CHECK-NEXT:   movaps  (%r{{[^,]*}}), %xmm{{.*}}
; CHECK-NEXT:   mulps   %xmm{{.*}}, %xmm{{.*}}
; CHECK-NEXT:   movaps  %xmm{{.*}}, (%r{{[^,]*}})
; CHECK-NEXT:   movaps  %xmm{{.*}}, 16(%r{{[^,]*}})
; CHECK-NEXT:   movaps  %xmm{{.*}}, 32(%r{{[^,]*}})
; CHECK-NEXT:   movaps  %xmm{{.*}}, 48(%r{{[^,]*}})
; CHECK-NEXT:   addps   %xmm{{.*}}, %xmm{{.*}}
; CHECK-NEXT:   addps   %xmm{{.*}}, %xmm{{.*}}
; CHECK-NEXT:   addps   %xmm{{.*}}, %xmm{{.*}}
; CHECK-NEXT:   addps   %xmm{{.*}}, %xmm{{.*}}
; CHECK-NEXT:   addq    $64, %r{{.*}}
; CHECK-NEXT:   addq    $64, %r{{.*}}
; CHECK-NEXT:   addq    $-16, %r{{.*}}
; CHECK-NEXT:   cmpq    $15, %r{{.*}}
; CHECK-NEXT:   jg

define void @test(ptr %arg, i64 %arg1, ptr nocapture %arg2, ptr nocapture %arg3, ptr %arg4, i64 %arg5, i64 %arg6) nounwind {
bb:
  %t = alloca float, align 4                      ; <ptr> [#uses=3]
  %t7 = alloca float, align 4                     ; <ptr> [#uses=2]
  %t8 = load float, ptr %arg3                         ; <float> [#uses=8]
  %t9 = ptrtoint ptr %arg to i64               ; <i64> [#uses=1]
  %t10 = ptrtoint ptr %arg4 to i64             ; <i64> [#uses=1]
  %t11 = xor i64 %t10, %t9                        ; <i64> [#uses=1]
  %t12 = and i64 %t11, 15                         ; <i64> [#uses=1]
  %t13 = icmp eq i64 %t12, 0                      ; <i1> [#uses=1]
  %t14 = xor i64 %arg1, 1                         ; <i64> [#uses=1]
  %t15 = xor i64 %arg5, 1                         ; <i64> [#uses=1]
  %t16 = or i64 %t15, %t14                        ; <i64> [#uses=1]
  %t17 = trunc i64 %t16 to i32                    ; <i32> [#uses=1]
  %t18 = icmp eq i32 %t17, 0                      ; <i1> [#uses=1]
  br i1 %t18, label %bb19, label %bb213

bb19:                                             ; preds = %bb
  %t20 = load float, ptr %arg2                        ; <float> [#uses=1]
  br label %bb21

bb21:                                             ; preds = %bb32, %bb19
  %t22 = phi i64 [ %t36, %bb32 ], [ 0, %bb19 ]    ; <i64> [#uses=21]
  %t23 = phi float [ %t35, %bb32 ], [ %t20, %bb19 ] ; <float> [#uses=6]
  %t24 = sub i64 %arg6, %t22                      ; <i64> [#uses=4]
  %t25 = getelementptr float, ptr %arg4, i64 %t22     ; <ptr> [#uses=4]
  %t26 = getelementptr float, ptr %arg, i64 %t22      ; <ptr> [#uses=3]
  %t27 = icmp sgt i64 %t24, 0                     ; <i1> [#uses=1]
  br i1 %t27, label %bb28, label %bb37

bb28:                                             ; preds = %bb21
  %t29 = ptrtoint ptr %t25 to i64              ; <i64> [#uses=1]
  %t30 = and i64 %t29, 15                         ; <i64> [#uses=1]
  %t31 = icmp eq i64 %t30, 0                      ; <i1> [#uses=1]
  br i1 %t31, label %bb37, label %bb32

bb32:                                             ; preds = %bb28
  %t33 = load float, ptr %t26                         ; <float> [#uses=1]
  %t34 = fmul float %t23, %t33                    ; <float> [#uses=1]
  store float %t34, ptr %t25
  %t35 = fadd float %t23, %t8                     ; <float> [#uses=1]
  %t36 = add i64 %t22, 1                          ; <i64> [#uses=1]
  br label %bb21

bb37:                                             ; preds = %bb28, %bb21
  %t38 = fmul float %t8, 4.000000e+00             ; <float> [#uses=1]
  store float %t38, ptr %t
  %t39 = fmul float %t8, 1.600000e+01             ; <float> [#uses=1]
  store float %t39, ptr %t7
  %t40 = fmul float %t8, 0.000000e+00             ; <float> [#uses=1]
  %t41 = fadd float %t23, %t40                    ; <float> [#uses=1]
  %t42 = insertelement <4 x float> undef, float %t41, i32 0 ; <<4 x float>> [#uses=1]
  %t43 = fadd float %t23, %t8                     ; <float> [#uses=1]
  %t44 = insertelement <4 x float> %t42, float %t43, i32 1 ; <<4 x float>> [#uses=1]
  %t45 = fmul float %t8, 2.000000e+00             ; <float> [#uses=1]
  %t46 = fadd float %t23, %t45                    ; <float> [#uses=1]
  %t47 = insertelement <4 x float> %t44, float %t46, i32 2 ; <<4 x float>> [#uses=1]
  %t48 = fmul float %t8, 3.000000e+00             ; <float> [#uses=1]
  %t49 = fadd float %t23, %t48                    ; <float> [#uses=1]
  %t50 = insertelement <4 x float> %t47, float %t49, i32 3 ; <<4 x float>> [#uses=5]
  %t51 = call <4 x float> asm "movss $1, $0\09\0Apshufd $$0, $0, $0", "=x,*m,~{dirflag},~{fpsr},~{flags}"(ptr %t) nounwind ; <<4 x float>> [#uses=3]
  %t52 = fadd <4 x float> %t50, %t51              ; <<4 x float>> [#uses=3]
  %t53 = fadd <4 x float> %t52, %t51              ; <<4 x float>> [#uses=3]
  %t54 = fadd <4 x float> %t53, %t51              ; <<4 x float>> [#uses=2]
  %t55 = call <4 x float> asm "movss $1, $0\09\0Apshufd $$0, $0, $0", "=x,*m,~{dirflag},~{fpsr},~{flags}"(ptr %t7) nounwind ; <<4 x float>> [#uses=8]
  %t56 = icmp sgt i64 %t24, 15                    ; <i1> [#uses=2]
  br i1 %t13, label %bb57, label %bb118

bb57:                                             ; preds = %bb37
  br i1 %t56, label %bb61, label %bb112

bb58:                                             ; preds = %bb68
  %t59 = getelementptr float, ptr %arg, i64 %t78      ; <ptr> [#uses=1]
  %t60 = getelementptr float, ptr %arg4, i64 %t78     ; <ptr> [#uses=1]
  br label %bb112

bb61:                                             ; preds = %bb57
  %t62 = add i64 %t22, 16                         ; <i64> [#uses=1]
  %t63 = add i64 %t22, 4                          ; <i64> [#uses=1]
  %t64 = add i64 %t22, 8                          ; <i64> [#uses=1]
  %t65 = add i64 %t22, 12                         ; <i64> [#uses=1]
  %t66 = add i64 %arg6, -16                       ; <i64> [#uses=1]
  %t67 = sub i64 %t66, %t22                       ; <i64> [#uses=1]
  br label %bb68

bb68:                                             ; preds = %bb68, %bb61
  %t69 = phi i64 [ 0, %bb61 ], [ %t111, %bb68 ]   ; <i64> [#uses=3]
  %t70 = phi <4 x float> [ %t54, %bb61 ], [ %t107, %bb68 ] ; <<4 x float>> [#uses=2]
  %t71 = phi <4 x float> [ %t50, %bb61 ], [ %t103, %bb68 ] ; <<4 x float>> [#uses=2]
  %t72 = phi <4 x float> [ %t53, %bb61 ], [ %t108, %bb68 ] ; <<4 x float>> [#uses=2]
  %t73 = phi <4 x float> [ %t52, %bb61 ], [ %t109, %bb68 ] ; <<4 x float>> [#uses=2]
  %t74 = shl i64 %t69, 4                          ; <i64> [#uses=5]
  %t75 = add i64 %t22, %t74                       ; <i64> [#uses=2]
  %t76 = getelementptr float, ptr %arg, i64 %t75      ; <ptr> [#uses=1]
  %t78 = add i64 %t62, %t74                       ; <i64> [#uses=2]
  %t79 = add i64 %t63, %t74                       ; <i64> [#uses=2]
  %t80 = getelementptr float, ptr %arg, i64 %t79      ; <ptr> [#uses=1]
  %t82 = add i64 %t64, %t74                       ; <i64> [#uses=2]
  %t83 = getelementptr float, ptr %arg, i64 %t82      ; <ptr> [#uses=1]
  %t85 = add i64 %t65, %t74                       ; <i64> [#uses=2]
  %t86 = getelementptr float, ptr %arg, i64 %t85      ; <ptr> [#uses=1]
  %t88 = getelementptr float, ptr %arg4, i64 %t75     ; <ptr> [#uses=1]
  %t90 = getelementptr float, ptr %arg4, i64 %t79     ; <ptr> [#uses=1]
  %t92 = getelementptr float, ptr %arg4, i64 %t82     ; <ptr> [#uses=1]
  %t94 = getelementptr float, ptr %arg4, i64 %t85     ; <ptr> [#uses=1]
  %t96 = mul i64 %t69, -16                        ; <i64> [#uses=1]
  %t97 = add i64 %t67, %t96                       ; <i64> [#uses=2]
  %t98 = load <4 x float>, ptr %t76                   ; <<4 x float>> [#uses=1]
  %t99 = load <4 x float>, ptr %t80                   ; <<4 x float>> [#uses=1]
  %t100 = load <4 x float>, ptr %t83                  ; <<4 x float>> [#uses=1]
  %t101 = load <4 x float>, ptr %t86                  ; <<4 x float>> [#uses=1]
  %t102 = fmul <4 x float> %t98, %t71             ; <<4 x float>> [#uses=1]
  %t103 = fadd <4 x float> %t71, %t55             ; <<4 x float>> [#uses=2]
  %t104 = fmul <4 x float> %t99, %t73             ; <<4 x float>> [#uses=1]
  %t105 = fmul <4 x float> %t100, %t72            ; <<4 x float>> [#uses=1]
  %t106 = fmul <4 x float> %t101, %t70            ; <<4 x float>> [#uses=1]
  store <4 x float> %t102, ptr %t88
  store <4 x float> %t104, ptr %t90
  store <4 x float> %t105, ptr %t92
  store <4 x float> %t106, ptr %t94
  %t107 = fadd <4 x float> %t70, %t55             ; <<4 x float>> [#uses=1]
  %t108 = fadd <4 x float> %t72, %t55             ; <<4 x float>> [#uses=1]
  %t109 = fadd <4 x float> %t73, %t55             ; <<4 x float>> [#uses=1]
  %t110 = icmp sgt i64 %t97, 15                   ; <i1> [#uses=1]
  %t111 = add i64 %t69, 1                         ; <i64> [#uses=1]
  br i1 %t110, label %bb68, label %bb58

bb112:                                            ; preds = %bb58, %bb57
  %t113 = phi ptr [ %t59, %bb58 ], [ %t26, %bb57 ] ; <ptr> [#uses=1]
  %t114 = phi ptr [ %t60, %bb58 ], [ %t25, %bb57 ] ; <ptr> [#uses=1]
  %t115 = phi <4 x float> [ %t103, %bb58 ], [ %t50, %bb57 ] ; <<4 x float>> [#uses=1]
  %t116 = phi i64 [ %t97, %bb58 ], [ %t24, %bb57 ] ; <i64> [#uses=1]
  %t117 = call <4 x float> asm "movss $1, $0\09\0Apshufd $$0, $0, $0", "=x,*m,~{dirflag},~{fpsr},~{flags}"(ptr %t) nounwind ; <<4 x float>> [#uses=0]
  br label %bb194

bb118:                                            ; preds = %bb37
  br i1 %t56, label %bb122, label %bb194

bb119:                                            ; preds = %bb137
  %t120 = getelementptr float, ptr %arg, i64 %t145    ; <ptr> [#uses=1]
  %t121 = getelementptr float, ptr %arg4, i64 %t145   ; <ptr> [#uses=1]
  br label %bb194

bb122:                                            ; preds = %bb118
  %t123 = add i64 %t22, -1                        ; <i64> [#uses=1]
  %t124 = getelementptr inbounds float, ptr %arg, i64 %t123 ; <ptr> [#uses=1]
  %t126 = load <4 x float>, ptr %t124                 ; <<4 x float>> [#uses=1]
  %t127 = add i64 %t22, 16                        ; <i64> [#uses=1]
  %t128 = add i64 %t22, 3                         ; <i64> [#uses=1]
  %t129 = add i64 %t22, 7                         ; <i64> [#uses=1]
  %t130 = add i64 %t22, 11                        ; <i64> [#uses=1]
  %t131 = add i64 %t22, 15                        ; <i64> [#uses=1]
  %t132 = add i64 %t22, 4                         ; <i64> [#uses=1]
  %t133 = add i64 %t22, 8                         ; <i64> [#uses=1]
  %t134 = add i64 %t22, 12                        ; <i64> [#uses=1]
  %t135 = add i64 %arg6, -16                      ; <i64> [#uses=1]
  %t136 = sub i64 %t135, %t22                     ; <i64> [#uses=1]
  br label %bb137

bb137:                                            ; preds = %bb137, %bb122
  %t138 = phi i64 [ 0, %bb122 ], [ %t193, %bb137 ] ; <i64> [#uses=3]
  %t139 = phi <4 x float> [ %t54, %bb122 ], [ %t189, %bb137 ] ; <<4 x float>> [#uses=2]
  %t140 = phi <4 x float> [ %t50, %bb122 ], [ %t185, %bb137 ] ; <<4 x float>> [#uses=2]
  %t141 = phi <4 x float> [ %t53, %bb122 ], [ %t190, %bb137 ] ; <<4 x float>> [#uses=2]
  %t142 = phi <4 x float> [ %t52, %bb122 ], [ %t191, %bb137 ] ; <<4 x float>> [#uses=2]
  %t143 = phi <4 x float> [ %t126, %bb122 ], [ %t175, %bb137 ] ; <<4 x float>> [#uses=1]
  %t144 = shl i64 %t138, 4                        ; <i64> [#uses=9]
  %t145 = add i64 %t127, %t144                    ; <i64> [#uses=2]
  %t146 = add i64 %t128, %t144                    ; <i64> [#uses=1]
  %t147 = getelementptr float, ptr %arg, i64 %t146    ; <ptr> [#uses=1]
  %t149 = add i64 %t129, %t144                    ; <i64> [#uses=1]
  %t150 = getelementptr float, ptr %arg, i64 %t149    ; <ptr> [#uses=1]
  %t152 = add i64 %t130, %t144                    ; <i64> [#uses=1]
  %t153 = getelementptr float, ptr %arg, i64 %t152    ; <ptr> [#uses=1]
  %t155 = add i64 %t131, %t144                    ; <i64> [#uses=1]
  %t156 = getelementptr float, ptr %arg, i64 %t155    ; <ptr> [#uses=1]
  %t158 = add i64 %t22, %t144                     ; <i64> [#uses=1]
  %t159 = getelementptr float, ptr %arg4, i64 %t158   ; <ptr> [#uses=1]
  %t161 = add i64 %t132, %t144                    ; <i64> [#uses=1]
  %t162 = getelementptr float, ptr %arg4, i64 %t161   ; <ptr> [#uses=1]
  %t164 = add i64 %t133, %t144                    ; <i64> [#uses=1]
  %t165 = getelementptr float, ptr %arg4, i64 %t164   ; <ptr> [#uses=1]
  %t167 = add i64 %t134, %t144                    ; <i64> [#uses=1]
  %t168 = getelementptr float, ptr %arg4, i64 %t167   ; <ptr> [#uses=1]
  %t170 = mul i64 %t138, -16                      ; <i64> [#uses=1]
  %t171 = add i64 %t136, %t170                    ; <i64> [#uses=2]
  %t172 = load <4 x float>, ptr %t147                 ; <<4 x float>> [#uses=2]
  %t173 = load <4 x float>, ptr %t150                 ; <<4 x float>> [#uses=2]
  %t174 = load <4 x float>, ptr %t153                 ; <<4 x float>> [#uses=2]
  %t175 = load <4 x float>, ptr %t156                 ; <<4 x float>> [#uses=2]
  %t176 = shufflevector <4 x float> %t143, <4 x float> %t172, <4 x i32> <i32 4, i32 1, i32 2, i32 3> ; <<4 x float>> [#uses=1]
  %t177 = shufflevector <4 x float> %t176, <4 x float> undef, <4 x i32> <i32 1, i32 2, i32 3, i32 0> ; <<4 x float>> [#uses=1]
  %t178 = shufflevector <4 x float> %t172, <4 x float> %t173, <4 x i32> <i32 4, i32 1, i32 2, i32 3> ; <<4 x float>> [#uses=1]
  %t179 = shufflevector <4 x float> %t178, <4 x float> undef, <4 x i32> <i32 1, i32 2, i32 3, i32 0> ; <<4 x float>> [#uses=1]
  %t180 = shufflevector <4 x float> %t173, <4 x float> %t174, <4 x i32> <i32 4, i32 1, i32 2, i32 3> ; <<4 x float>> [#uses=1]
  %t181 = shufflevector <4 x float> %t180, <4 x float> undef, <4 x i32> <i32 1, i32 2, i32 3, i32 0> ; <<4 x float>> [#uses=1]
  %t182 = shufflevector <4 x float> %t174, <4 x float> %t175, <4 x i32> <i32 4, i32 1, i32 2, i32 3> ; <<4 x float>> [#uses=1]
  %t183 = shufflevector <4 x float> %t182, <4 x float> undef, <4 x i32> <i32 1, i32 2, i32 3, i32 0> ; <<4 x float>> [#uses=1]
  %t184 = fmul <4 x float> %t177, %t140           ; <<4 x float>> [#uses=1]
  %t185 = fadd <4 x float> %t140, %t55            ; <<4 x float>> [#uses=2]
  %t186 = fmul <4 x float> %t179, %t142           ; <<4 x float>> [#uses=1]
  %t187 = fmul <4 x float> %t181, %t141           ; <<4 x float>> [#uses=1]
  %t188 = fmul <4 x float> %t183, %t139           ; <<4 x float>> [#uses=1]
  store <4 x float> %t184, ptr %t159
  store <4 x float> %t186, ptr %t162
  store <4 x float> %t187, ptr %t165
  store <4 x float> %t188, ptr %t168
  %t189 = fadd <4 x float> %t139, %t55            ; <<4 x float>> [#uses=1]
  %t190 = fadd <4 x float> %t141, %t55            ; <<4 x float>> [#uses=1]
  %t191 = fadd <4 x float> %t142, %t55            ; <<4 x float>> [#uses=1]
  %t192 = icmp sgt i64 %t171, 15                  ; <i1> [#uses=1]
  %t193 = add i64 %t138, 1                        ; <i64> [#uses=1]
  br i1 %t192, label %bb137, label %bb119

bb194:                                            ; preds = %bb119, %bb118, %bb112
  %t195 = phi i64 [ %t116, %bb112 ], [ %t171, %bb119 ], [ %t24, %bb118 ] ; <i64> [#uses=2]
  %t196 = phi <4 x float> [ %t115, %bb112 ], [ %t185, %bb119 ], [ %t50, %bb118 ] ; <<4 x float>> [#uses=1]
  %t197 = phi ptr [ %t114, %bb112 ], [ %t121, %bb119 ], [ %t25, %bb118 ] ; <ptr> [#uses=1]
  %t198 = phi ptr [ %t113, %bb112 ], [ %t120, %bb119 ], [ %t26, %bb118 ] ; <ptr> [#uses=1]
  %t199 = extractelement <4 x float> %t196, i32 0 ; <float> [#uses=2]
  %t200 = icmp sgt i64 %t195, 0                   ; <i1> [#uses=1]
  br i1 %t200, label %bb201, label %bb211

bb201:                                            ; preds = %bb201, %bb194
  %t202 = phi i64 [ %t209, %bb201 ], [ 0, %bb194 ] ; <i64> [#uses=3]
  %t203 = phi float [ %t208, %bb201 ], [ %t199, %bb194 ] ; <float> [#uses=2]
  %t204 = getelementptr float, ptr %t198, i64 %t202   ; <ptr> [#uses=1]
  %t205 = getelementptr float, ptr %t197, i64 %t202   ; <ptr> [#uses=1]
  %t206 = load float, ptr %t204                       ; <float> [#uses=1]
  %t207 = fmul float %t203, %t206                 ; <float> [#uses=1]
  store float %t207, ptr %t205
  %t208 = fadd float %t203, %t8                   ; <float> [#uses=2]
  %t209 = add i64 %t202, 1                        ; <i64> [#uses=2]
  %t210 = icmp eq i64 %t209, %t195                ; <i1> [#uses=1]
  br i1 %t210, label %bb211, label %bb201

bb211:                                            ; preds = %bb201, %bb194
  %t212 = phi float [ %t199, %bb194 ], [ %t208, %bb201 ] ; <float> [#uses=1]
  store float %t212, ptr %arg2
  ret void

bb213:                                            ; preds = %bb
  ret void
}
