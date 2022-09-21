; RUN: opt -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -keep-loops=false -S < %s | FileCheck %s
; RUN: opt -passes='simplifycfg<no-keep-loops>' -S < %s | FileCheck %s

define void @test1(i32 %n) #0 {
entry:
  %n.addr = alloca i32, align 4
  %count = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  store i32 0, ptr %count, align 4
  br label %while.cond

while.cond:                                       ; preds = %if.end, %entry
  %0 = load i32, ptr %count, align 4
  %1 = load i32, ptr %n.addr, align 4
  %cmp = icmp ule i32 %0, %1
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %2 = load i32, ptr %count, align 4
  %rem = urem i32 %2, 2
  %cmp1 = icmp eq i32 %rem, 0
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %while.body
  %3 = load i32, ptr %count, align 4
  %add = add i32 %3, 1
  store i32 %add, ptr %count, align 4
  br label %if.end

; CHECK: if.then:
; CHECK:  br label %while.cond, !llvm.loop !0

if.else:                                          ; preds = %while.body
  %4 = load i32, ptr %count, align 4
  %add2 = add i32 %4, 2
  store i32 %add2, ptr %count, align 4
  br label %if.end

; CHECK: if.else:
; CHECK:  br label %while.cond, !llvm.loop !0

if.end:                                           ; preds = %if.else, %if.then
  br label %while.cond, !llvm.loop !0

while.end:                                        ; preds = %while.cond
  ret void
}

; The test case is constructed based on the following C++ code,
; as a simplified test case to show why `llvm.loop.unroll.enable`
; could be dropped.
;
; While the C++ code itself might have the inner-loop unrolled (e.g., with -O3),
; the loss of inner-loop unroll metadata is a bug.
; Under some optimization pipelines (e.g., FullLoopUnroll pass is skipped in ThinLTO prelink stage),
; and in real-world C++ code (e.g., with larger loop body), failing to
; preserve loop unroll metadata could cause missed loop unroll.
;
; constexpr int kUnroll = 5;
; int sum(int a, int b, int step, const int remainder, int* input) {
;    int i = a, j = b;
;    int sum = 0;
;    while(j - i > remainder) {
;        i += step;
;        #pragma unroll
;        for (int k = 0; k < kUnroll; k++) {
;           asm volatile ("add %w0, %w1\n" : "=r"(sum) : "r"(input[k + i]):"cc");
;        }
;    }
;    return sum;
; }
define i32 @test2(i32 %a, i32 %b, i32 %step, i32 %remainder, ptr %input) {
entry:
  br label %while.cond

while.cond.loopexit:                              ; preds = %for.body
  br label %while.cond, !llvm.loop !2

while.cond:                                       ; preds = %while.cond.loopexit, %entry
  %i.0 = phi i32 [ %a, %entry ], [ %add, %while.cond.loopexit ]
  %sum.0 = phi i32 [ 0, %entry ], [ %1, %while.cond.loopexit ]
  %sub = sub nsw i32 %b, %i.0
  %cmp = icmp sgt i32 %sub, %remainder
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %add = add nsw i32 %i.0, %step
  br label %for.body

for.body:                                         ; preds = %while.body, %for.body
  %k.07 = phi i32 [ 0, %while.body ], [ %inc, %for.body ]
  %add2 = add nsw i32 %k.07, %add
  %idxprom = sext i32 %add2 to i64
  %arrayidx = getelementptr inbounds i32, ptr %input, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  %1 = tail call i32 asm sideeffect "add ${0:w}, ${1:w}\0A", "=r,r,~{cc}"(i32 %0)
  %inc = add nuw nsw i32 %k.07, 1
  %cmp1 = icmp ult i32 %inc, 5
  br i1 %cmp1, label %for.body, label %while.cond.loopexit, !llvm.loop !4

while.end:                                        ; preds = %while.cond
  %sum.0.lcssa = phi i32 [ %sum.0, %while.cond ]
  ret i32 %sum.0.lcssa
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.distribute.enable", i1 true}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.mustprogress"}
!4 = distinct !{!4, !3, !5}
!5 = !{!"llvm.loop.unroll.enable"}
; CHECK: !0 = distinct !{!0, !1}
; CHECK: !1 = !{!"llvm.loop.distribute.enable", i1 true}
; CHECK-NOT: !{!"llvm.loop.unroll.enable"}
