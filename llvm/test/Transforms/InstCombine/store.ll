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

define void @store_at_gep_off_null(i64 %offset) {
; CHECK-LABEL: @store_at_gep_off_null
; CHECK: store i32 undef, i32* %ptr
   %ptr = getelementptr i32, i32 *null, i64 %offset
   store i32 24, i32* %ptr
   ret void
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

define void @dse1(i32* %p) {
; CHECK-LABEL: dse1
; CHECK-NEXT: store
; CHECK-NEXT: ret
  store i32 0, i32* %p
  store i32 0, i32* %p
  ret void
} 

; Slightly subtle: if we're mixing atomic and non-atomic access to the
; same location, then the contents of the location are undefined if there's
; an actual race.  As such, we're free to pick either store under the 
; assumption that we're not racing with any other thread.
define void @dse2(i32* %p) {
; CHECK-LABEL: dse2
; CHECK-NEXT: store i32 0, i32* %p
; CHECK-NEXT: ret
  store atomic i32 0, i32* %p unordered, align 4
  store i32 0, i32* %p
  ret void
} 

define void @dse3(i32* %p) {
; CHECK-LABEL: dse3
; CHECK-NEXT: store atomic i32 0, i32* %p unordered, align 4
; CHECK-NEXT: ret
  store i32 0, i32* %p
  store atomic i32 0, i32* %p unordered, align 4
  ret void
} 

define void @dse4(i32* %p) {
; CHECK-LABEL: dse4
; CHECK-NEXT: store atomic i32 0, i32* %p unordered, align 4
; CHECK-NEXT: ret
  store atomic i32 0, i32* %p unordered, align 4
  store atomic i32 0, i32* %p unordered, align 4
  ret void
} 

; Implementation limit - could remove unordered store here, but
; currently don't.
define void @dse5(i32* %p) {
; CHECK-LABEL: dse5
; CHECK-NEXT: store
; CHECK-NEXT: store
; CHECK-NEXT: ret
  store atomic i32 0, i32* %p unordered, align 4
  store atomic i32 0, i32* %p seq_cst, align 4
  ret void
}

define void @write_back1(i32* %p) {
; CHECK-LABEL: write_back1
; CHECK-NEXT: ret
  %v = load i32, i32* %p
  store i32 %v, i32* %p
  ret void
} 

define void @write_back2(i32* %p) {
; CHECK-LABEL: write_back2
; CHECK-NEXT: ret
  %v = load atomic i32, i32* %p unordered, align 4
  store i32 %v, i32* %p
  ret void
} 

define void @write_back3(i32* %p) {
; CHECK-LABEL: write_back3
; CHECK-NEXT: ret
  %v = load i32, i32* %p
  store atomic i32 %v, i32* %p unordered, align 4
  ret void
} 

define void @write_back4(i32* %p) {
; CHECK-LABEL: write_back4
; CHECK-NEXT: ret
  %v = load atomic i32, i32* %p unordered, align 4
  store atomic i32 %v, i32* %p unordered, align 4
  ret void
} 

; Can't remove store due to ordering side effect
define void @write_back5(i32* %p) {
; CHECK-LABEL: write_back5
; CHECK-NEXT: load
; CHECK-NEXT: store
; CHECK-NEXT: ret
  %v = load atomic i32, i32* %p unordered, align 4
  store atomic i32 %v, i32* %p seq_cst, align 4
  ret void
}

define void @write_back6(i32* %p) {
; CHECK-LABEL: write_back6
; CHECK-NEXT: load
; CHECK-NEXT: ret
  %v = load atomic i32, i32* %p seq_cst, align 4
  store atomic i32 %v, i32* %p unordered, align 4
  ret void
}

define void @write_back7(i32* %p) {
; CHECK-LABEL: write_back7
; CHECK-NEXT: load
; CHECK-NEXT: ret
  %v = load atomic volatile i32, i32* %p seq_cst, align 4
  store atomic i32 %v, i32* %p unordered, align 4
  ret void
}

!0 = !{!4, !4, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!"float", !1}
!4 = !{!"int", !1}
