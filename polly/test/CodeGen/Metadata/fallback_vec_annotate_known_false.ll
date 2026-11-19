; RUN: opt %loadNPMPolly '-passes=polly-custom<prune;codegen>' \
; RUN:   -plugin-arg=Polly,-polly-process-unprofitable=false -verify-each -S < %s | \
; RUN:   FileCheck %s --check-prefix=PRUNED --implicit-check-not=llvm.loop.vectorize.disable
; RUN: opt %loadNPMPolly '-passes=polly-custom<prune;codegen>' \
; RUN:   -plugin-arg=Polly,-polly-process-unprofitable=true -verify-each -S < %s | \
; RUN:   FileCheck %s --check-prefix=DYNAMIC --implicit-check-not='and i1 false,'
;
; Polly constructs a SCoP from combining the two loops in the code below.
; Pruning invalidates this SCoP, but its alias checks leave the RTC as an
; instruction containing "and i1 false, ...". computeKnownBits proves it false,
; preserving fallback vectorization.
;
; Equivalent C, compiled with -ffast-math.
;
; #include <stdint.h>
;
; void normalize(float *a, float *b, int64_t n) {
;     if (n <= 0)
;         return;
;
;     float sum = 0;
;     for (int64_t i = 0; i < n; ++i)
;         sum += a[i];
;
;     for (int64_t j = 0; j < n; ++j)
;         b[j] = a[j] / sum;
; }
;
; PRUNED-LABEL: define void @normalize(
; PRUNED: [[FALSE:%[a-zA-Z0-9._]+]] = and i1 false, %{{.*}}
; PRUNED-NEXT: %polly.rtc.result = and i1 [[FALSE]], {{.*}}
; PRUNED-NEXT: br i1 %polly.rtc.result, label %polly.start,
; PRUNED: polly.start:
;
; With pruning disabled, the RTC remains dynamic and fallback vectorization
; should still be disabled.
;
; DYNAMIC-LABEL: define void @normalize(
; DYNAMIC: br i1 %polly.rtc.result, label %polly.start,
; DYNAMIC: !{!"llvm.loop.vectorize.disable"}

define void @normalize(ptr %a, ptr %b, i64 %n) {
entry:
  %nonempty = icmp sgt i64 %n, 0
  br i1 %nonempty, label %sum.preheader, label %exit

sum.preheader:
  br label %sum.loop

sum.loop:
  %i = phi i64 [ 0, %sum.preheader ], [ %i.next, %sum.loop ]
  %sum = phi float [ 0.0, %sum.preheader ], [ %sum.next, %sum.loop ]
  %a.i = getelementptr inbounds float, ptr %a, i64 %i
  %value = load float, ptr %a.i, align 4
  %sum.next = fadd fast float %sum, %value
  %i.next = add nuw nsw i64 %i, 1
  %sum.done = icmp eq i64 %i.next, %n
  br i1 %sum.done, label %write.preheader, label %sum.loop

write.preheader:
  %total = phi float [ %sum.next, %sum.loop ]
  br label %write.loop

write.loop:
  %j = phi i64 [ 0, %write.preheader ], [ %j.next, %write.loop ]
  %a.j = getelementptr inbounds float, ptr %a, i64 %j
  %input = load float, ptr %a.j, align 4
  %scaled = fdiv fast float %input, %total
  %b.j = getelementptr inbounds float, ptr %b, i64 %j
  store float %scaled, ptr %b.j, align 4
  %j.next = add nuw nsw i64 %j, 1
  %write.done = icmp eq i64 %j.next, %n
  br i1 %write.done, label %exit, label %write.loop

exit:
  ret void
}
