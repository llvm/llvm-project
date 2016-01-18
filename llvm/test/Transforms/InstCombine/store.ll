; RUN: opt < %s -instcombine -S | FileCheck %s

define void @test1(i32* %P) {
        store i32 undef, i32* %P
        store i32 123, i32* undef
        store i32 124, i32* null
        ret void
; CHECK-LABEL: @test1(
; CHECK-NEXT: store i32 123, i32* undef
; CHECK-NEXT: store i32 undef, i32* null
; CHECK-NEXT: ret void
}

define void @test2(i32* %P) {
        %X = load i32, i32* %P               ; <i32> [#uses=1]
        %Y = add i32 %X, 0              ; <i32> [#uses=1]
        store i32 %Y, i32* %P
        ret void
; CHECK-LABEL: @test2(
; CHECK-NEXT: ret void
}

;; Simple sinking tests

; "if then else"
define i32 @test3(i1 %C) {
	%A = alloca i32
        br i1 %C, label %Cond, label %Cond2

Cond:
        store i32 -987654321, i32* %A
        br label %Cont

Cond2:
	store i32 47, i32* %A
	br label %Cont

Cont:
	%V = load i32, i32* %A
	ret i32 %V
; CHECK-LABEL: @test3(
; CHECK-NOT: alloca
; CHECK: Cont:
; CHECK-NEXT:  %storemerge = phi i32 [ -987654321, %Cond ], [ 47, %Cond2 ]
; CHECK-NEXT:  ret i32 %storemerge
}

; "if then"
define i32 @test4(i1 %C) {
	%A = alloca i32
	store i32 47, i32* %A
        br i1 %C, label %Cond, label %Cont

Cond:
        store i32 -987654321, i32* %A
        br label %Cont

Cont:
	%V = load i32, i32* %A
	ret i32 %V
; CHECK-LABEL: @test4(
; CHECK-NOT: alloca
; CHECK: Cont:
; CHECK-NEXT:  %storemerge = phi i32 [ -987654321, %Cond ], [ 47, %0 ]
; CHECK-NEXT:  ret i32 %storemerge
}

; "if then"
define void @test5(i1 %C, i32* %P) {
	store i32 47, i32* %P, align 1
        br i1 %C, label %Cond, label %Cont

Cond:
        store i32 -987654321, i32* %P, align 1
        br label %Cont

Cont:
	ret void
; CHECK-LABEL: @test5(
; CHECK: Cont:
; CHECK-NEXT:  %storemerge = phi i32
; CHECK-NEXT:  store i32 %storemerge, i32* %P, align 1
; CHECK-NEXT:  ret void
}


; PR14753 - merging two stores should preserve the TBAA tag.
define void @test6(i32 %n, float* %a, i32* %gi) nounwind uwtable ssp {
entry:
  store i32 42, i32* %gi, align 4, !tbaa !0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %storemerge = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = load i32, i32* %gi, align 4, !tbaa !0
  %cmp = icmp slt i32 %0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %idxprom = sext i32 %0 to i64
  %arrayidx = getelementptr inbounds float, float* %a, i64 %idxprom
  store float 0.000000e+00, float* %arrayidx, align 4, !tbaa !3
  %1 = load i32, i32* %gi, align 4, !tbaa !0
  %inc = add nsw i32 %1, 1
  store i32 %inc, i32* %gi, align 4, !tbaa !0
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
; CHECK-LABEL: @test6(
; CHECK: for.cond:
; CHECK-NEXT: phi i32 [ 42
; CHECK-NEXT: store i32 %storemerge, i32* %gi, align 4, !tbaa !0
}

!0 = !{!4, !4, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!"float", !1}
!4 = !{!"int", !1}
