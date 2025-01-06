; RUN: opt -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -keep-loops=false -S < %s | FileCheck %s
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
; CHECK:  br label %while.cond, !llvm.loop !1

if.else:                                          ; preds = %while.body
  %4 = load i32, ptr %count, align 4
  %add2 = add i32 %4, 2
  store i32 %add2, ptr %count, align 4
  br label %if.end

; CHECK: if.else:
; CHECK:  br label %while.cond, !llvm.loop !1

if.end:                                           ; preds = %if.else, %if.then
  br label %while.cond, !llvm.loop !1

while.end:                                        ; preds = %while.cond
  ret void
}

; Test that empty loop latch `while.cond.loopexit` will not be folded into its successor if its
; predecessor blocks are also loop latches.
;
; The test case is constructed based on the following C++ code.
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
  br label %while.cond, !llvm.loop !3

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
  br i1 %cmp1, label %for.body, label %while.cond.loopexit, !llvm.loop !5

while.end:                                        ; preds = %while.cond
  %sum.0.lcssa = phi i32 [ %sum.0, %while.cond ]
  ret i32 %sum.0.lcssa
}

; Test that the condition tested above does not trigger when the loop metadata consists only of debug locations,
; i.e.the empty loop latch `while.cond.loopexit` *will* be folded into its successor if its
; predecessor blocks are also loop latches and any loop metadata attached to it consists of debug information.
;
define i32 @test3(i32 %a, i32 %b, i32 %step, i32 %remainder, ptr %input) !dbg !7 {
entry:
  br label %while.cond

;CHECK-LABEL: @test3( 
;CHECK-NOT: while.cond.loopexit
while.cond.loopexit:                              ; preds = %for.body
  br label %while.cond, !llvm.loop !10

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
  br i1 %cmp1, label %for.body, label %while.cond.loopexit, !llvm.loop !5

while.end:                                        ; preds = %while.cond
  %sum.0.lcssa = phi i32 [ %sum.0, %while.cond ]
  ret i32 %sum.0.lcssa
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.distribute.enable", i1 true}
!3 = distinct !{!3, !4}
!4 = !{!"llvm.loop.mustprogress"}
!5 = distinct !{!5, !4, !6}
!6 = !{!"llvm.loop.unroll.enable"}
!7 = distinct !DISubprogram(name: "test3", scope: !8, file: !8, spFlags: DISPFlagDefinition, unit: !9)
!8 = !DIFile(filename: "preserve-llvm-loop-metadata.ll", directory: "/")
!9 = distinct !DICompileUnit(language: DW_LANG_C99, file: !8, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!10 = distinct !{!10, !11, !13}
!11 = !DILocation(line: 8, column: 4, scope: !12)
!12 = distinct !DILexicalBlock(scope: !7, file: !8, line: 8, column: 2)
!13 = !DILocation(line: 9, column: 23, scope: !12)

; CHECK: !1 = distinct !{!1, !2}
; CHECK: !2 = !{!"llvm.loop.distribute.enable", i1 true}
; CHECK: !3 = distinct !{!3, !4}
; CHECK: !4 = !{!"llvm.loop.mustprogress"}
; CHECK: !5 = distinct !{!5, !4, !6}
; CHECK: !6 = !{!"llvm.loop.unroll.enable"}
; CHECK-NOT: !10 = distinct !{!10, !11, !13}
