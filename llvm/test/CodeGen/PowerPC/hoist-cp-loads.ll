; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O3 \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   -stop-before=greedy < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O3 \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   -disable-aggressive-cpl-hoist \
; RUN:   -stop-before=greedy < %s | FileCheck %s --check-prefix=NOHOIST

;; This test checks that when register pressure is high, -disable-aggressive-cpl-hoist will control
;; the hoist of the constant loads. 
;; There are 34 PLXVpc LICM candicates, the first 32 PLXVpc will always be hoisted
;; because the register pressure is low. But the left 2 PLXVpc will be controled
;; by -disable-aggressive-cpl-hoist option. By default, these 2 PLXVpc will be hoisted too.
;; With -disable-aggressive-cpl-hoist, these 2 PLXVpc will be kept inside the loop.

; CHECK:   name: test
; CHECK:         for.body.preheader:
; CHECK:           MTCTR8loop
; CHECK-COUNT:     PLXVpc 34
; CHECK:           B %bb.
; CHECK:         for.body:
; CHECK-NO:        PLXVpc
;
; NOHOIST:   name: test
; NOHOIST:         for.body.preheader:
; NOHOIST:           MTCTR8loop
; NOHOIST-COUNT:     PLXVpc 32
; NOHOIST:           B %bb.
; NOHOIST:         for.body:
; NOHOIST-COUNT:     PLXVpc 2

define void @test(ptr %array, ptr readonly %indicate, i32 signext %count) nounwind {
entry:
  %cmp24 = icmp sgt i32 %count, 0
  br i1 %cmp24, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext nneg i32 %count to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.inc
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.inc ]
  %arrayidx4 = getelementptr inbounds i8, ptr %indicate, i64 %indvars.iv
  %ind = load i8, ptr %arrayidx4, align 1
  %cmp1 = icmp ugt i8 %ind, 10
  %arrayidx = getelementptr inbounds i8, ptr %array, i64 %indvars.iv
  %0 = load <16 x i16>, ptr %arrayidx, align 1
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %add = add <16 x i16> %0, <i16 1, i16 2, i16 3, i16 4, i16 5,i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 1>
  store volatile <16 x i16> %add, ptr %arrayidx, align 32
  %add2 = add <16 x i16> %0, <i16 2, i16 2, i16 3, i16 4, i16 5,i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 2>
  store volatile <16 x i16> %add2, ptr %arrayidx, align 32
  %add3 = add <16 x i16> %0, <i16 3, i16 2, i16 3, i16 4, i16 5,i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 3>
  store volatile <16 x i16> %add3, ptr %arrayidx, align 32
  %add4 = add <16 x i16> %0, <i16 4, i16 2, i16 3, i16 4, i16 5,i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 4>
  store volatile <16 x i16> %add4, ptr %arrayidx, align 32
  %add5 = add <16 x i16> %0, <i16 5, i16 2, i16 3, i16 4, i16 5,i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 5>
  store volatile <16 x i16> %add5, ptr %arrayidx, align 32
  %add6 = add <16 x i16> %0, <i16 6, i16 2, i16 3, i16 4, i16 5,i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 6>
  store volatile <16 x i16> %add6, ptr %arrayidx, align 32
  %add7 = add <16 x i16> %0, <i16 7, i16 2, i16 3, i16 4, i16 5,i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 7>
  store volatile <16 x i16> %add7, ptr %arrayidx, align 32
  %add8 = add <16 x i16> %0, <i16 8, i16 2, i16 3, i16 4, i16 5,i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 8>
  store volatile <16 x i16> %add8, ptr %arrayidx, align 32
  %add9 = add <16 x i16> %0, <i16 9, i16 2, i16 3, i16 4, i16 5,i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 9>
  store volatile <16 x i16> %add9, ptr %arrayidx, align 32
  br label %for.inc

if.else:                                          ; preds = %for.body
  %add10 = add <16 x i16> %0, <i16 10, i16 2, i16 3, i16 4, i16 5,i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 10>
  store volatile <16 x i16> %add10, ptr %arrayidx, align 32
  %add11 = add <16 x i16> %0, <i16 11, i16 2, i16 3, i16 4, i16 5,i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 11>
  store volatile <16 x i16> %add11, ptr %arrayidx, align 32
  %add12 = add <16 x i16> %0, <i16 12, i16 2, i16 3, i16 4, i16 5,i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 12>
  store volatile <16 x i16> %add12, ptr %arrayidx, align 32
  %add13 = add <16 x i16> %0, <i16 13, i16 2, i16 3, i16 4, i16 5,i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 13>
  store volatile <16 x i16> %add13, ptr %arrayidx, align 32
  %add14 = add <16 x i16> %0, <i16 14, i16 2, i16 3, i16 4, i16 5,i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 14>
  store volatile <16 x i16> %add14, ptr %arrayidx, align 32
  %add15 = add <16 x i16> %0, <i16 15, i16 2, i16 3, i16 4, i16 5,i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 15>
  store volatile <16 x i16> %add15, ptr %arrayidx, align 32
  %add16 = add <16 x i16> %0, <i16 16, i16 2, i16 3, i16 4, i16 5,i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 16>
  store volatile <16 x i16> %add16, ptr %arrayidx, align 32
  %add17 = add <16 x i16> %0, <i16 17, i16 2, i16 3, i16 4, i16 5,i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 17>
  store volatile <16 x i16> %add17, ptr %arrayidx, align 32

  br label %for.inc

for.inc:                                          ; preds = %if.then, %if.else
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
