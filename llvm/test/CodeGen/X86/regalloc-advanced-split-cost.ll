; RUN: llc < %s -march=x86 -regalloc=greedy --debug-only=regalloc 2>&1 | FileCheck %s

; REQUIRES: asserts

; This test is meant to make sure that the weight of local intervals that are
; created during split is taken into account when choosing the best candidate
; register.
; %shl is the interval that will be split.
; The inline assembly calls interfere with %shl and make only 2 available split
; candidates - %esi and %ebp.
; The old code would have chosen %esi as the split candidate ignoring the fact
; that this choice will cause the creation of a local interval that will have a
;  certain spill cost.
; The new code choses %ebp as the split candidate as it has lower spill cost.

; Make sure the split behaves as expected
; CHECK: RS_Split Cascade 1
; CHECK-NOT: $eax	static = 
; CHECK: $eax	no positive bundles
; CHECK-NEXT: $ecx	no positive bundles
; CHECK-NEXT: $edx	no positive bundles
; CHECK-NEXT: $esi	static = 
; CHECK-NEXT: $edi	no positive bundles
; CHECK-NEXT: $ebx	no positive bundles
; CHECK-NEXT: $ebp	static = 
; CHECK: Split for $ebp

; Function Attrs: nounwind
define i32 @foo(i32* %array, i32 %cond1, i32 %val) local_unnamed_addr #0 {
entry:
  %array.addr = alloca i32*, align 4
  store i32* %array, i32** %array.addr, align 4, !tbaa !3
  %0 = load i32, i32* %array, align 4, !tbaa !7
  %arrayidx1 = getelementptr inbounds i32, i32* %array, i32 1
  %1 = load i32, i32* %arrayidx1, align 4, !tbaa !7
  %arrayidx2 = getelementptr inbounds i32, i32* %array, i32 2
  %2 = load i32, i32* %arrayidx2, align 4, !tbaa !7
  %arrayidx3 = getelementptr inbounds i32, i32* %array, i32 3
  %3 = load i32, i32* %arrayidx3, align 4, !tbaa !7
  %arrayidx4 = getelementptr inbounds i32, i32* %array, i32 4
  %4 = load i32, i32* %arrayidx4, align 4, !tbaa !7
  %arrayidx6 = getelementptr inbounds i32, i32* %array, i32 %val
  %5 = load i32, i32* %arrayidx6, align 4, !tbaa !7
  %shl = shl i32 %5, 5
  %tobool = icmp eq i32 %cond1, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %arrayidx7 = getelementptr inbounds i32, i32* %array, i32 6
  store i32 %shl, i32* %arrayidx7, align 4, !tbaa !7
  call void asm "nop", "=*m,r,r,r,r,r,*m,~{dirflag},~{fpsr},~{flags}"(i32** nonnull %array.addr, i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32** nonnull %array.addr) #1, !srcloc !9
  %6 = load i32*, i32** %array.addr, align 4, !tbaa !3
  %arrayidx8 = getelementptr inbounds i32, i32* %6, i32 7
  br label %if.end

if.else:                                          ; preds = %entry
  %arrayidx5 = getelementptr inbounds i32, i32* %array, i32 5
  %7 = load i32, i32* %arrayidx5, align 4, !tbaa !7
  %arrayidx9 = getelementptr inbounds i32, i32* %array, i32 8
  store i32 %shl, i32* %arrayidx9, align 4, !tbaa !7
  call void asm "nop", "=*m,{ax},{bx},{cx},{dx},{di},{si},{ebp},*m,~{dirflag},~{fpsr},~{flags}"(i32** nonnull %array.addr, i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %7, i32* undef, i32** nonnull %array.addr) #1, !srcloc !10
  %8 = load i32*, i32** %array.addr, align 4, !tbaa !3
  %arrayidx10 = getelementptr inbounds i32, i32* %8, i32 9
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %arrayidx10.sink = phi i32* [ %arrayidx10, %if.else ], [ %arrayidx8, %if.then ]
  %9 = phi i32* [ %8, %if.else ], [ %6, %if.then ]
  store i32 %shl, i32* %arrayidx10.sink, align 4, !tbaa !7
  %10 = load i32, i32* %9, align 4, !tbaa !7
  %add = add nsw i32 %10, %shl
  ret i32 %add
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"NumRegisterParameters", i32 0}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{!"clang version 6.0.0"}
!3 = !{!4, !4, i64 0}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !5, i64 0}
!9 = !{i32 268}
!10 = !{i32 390}
