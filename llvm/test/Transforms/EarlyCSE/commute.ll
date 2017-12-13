; RUN: opt < %s -S -early-cse | FileCheck %s
; RUN: opt < %s -S -basicaa -early-cse-memssa | FileCheck %s

define void @test1(float %A, float %B, float* %PA, float* %PB) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:    [[C:%.*]] = fadd float %A, %B
; CHECK-NEXT:    store float [[C]], float* %PA
; CHECK-NEXT:    store float [[C]], float* %PB
; CHECK-NEXT:    ret void
;
  %C = fadd float %A, %B
  store float %C, float* %PA
  %D = fadd float %B, %A
  store float %D, float* %PB
  ret void
}

define void @test2(float %A, float %B, i1* %PA, i1* %PB) {
; CHECK-LABEL: @test2(
; CHECK-NEXT:    [[C:%.*]] = fcmp oeq float %A, %B
; CHECK-NEXT:    store i1 [[C]], i1* %PA
; CHECK-NEXT:    store i1 [[C]], i1* %PB
; CHECK-NEXT:    ret void
;
  %C = fcmp oeq float %A, %B
  store i1 %C, i1* %PA
  %D = fcmp oeq float %B, %A
  store i1 %D, i1* %PB
  ret void
}

define void @test3(float %A, float %B, i1* %PA, i1* %PB) {
; CHECK-LABEL: @test3(
; CHECK-NEXT:    [[C:%.*]] = fcmp uge float %A, %B
; CHECK-NEXT:    store i1 [[C]], i1* %PA
; CHECK-NEXT:    store i1 [[C]], i1* %PB
; CHECK-NEXT:    ret void
;
  %C = fcmp uge float %A, %B
  store i1 %C, i1* %PA
  %D = fcmp ule float %B, %A
  store i1 %D, i1* %PB
  ret void
}

define void @test4(i32 %A, i32 %B, i1* %PA, i1* %PB) {
; CHECK-LABEL: @test4(
; CHECK-NEXT:    [[C:%.*]] = icmp eq i32 %A, %B
; CHECK-NEXT:    store i1 [[C]], i1* %PA
; CHECK-NEXT:    store i1 [[C]], i1* %PB
; CHECK-NEXT:    ret void
;
  %C = icmp eq i32 %A, %B
  store i1 %C, i1* %PA
  %D = icmp eq i32 %B, %A
  store i1 %D, i1* %PB
  ret void
}

define void @test5(i32 %A, i32 %B, i1* %PA, i1* %PB) {
; CHECK-LABEL: @test5(
; CHECK-NEXT:    [[C:%.*]] = icmp sgt i32 %A, %B
; CHECK-NEXT:    store i1 [[C]], i1* %PA
; CHECK-NEXT:    store i1 [[C]], i1* %PB
; CHECK-NEXT:    ret void
;
  %C = icmp sgt i32 %A, %B
  store i1 %C, i1* %PA
  %D = icmp slt i32 %B, %A
  store i1 %D, i1* %PB
  ret void
}

; Min/max operands may be commuted in the compare and select.

define i8 @smin_commute(i8 %a, i8 %b) {
; CHECK-LABEL: @smin_commute(
; CHECK-NEXT:    [[CMP1:%.*]] = icmp slt i8 %a, %b
; CHECK-NEXT:    [[CMP2:%.*]] = icmp slt i8 %b, %a
; CHECK-NEXT:    [[M1:%.*]] = select i1 [[CMP1]], i8 %a, i8 %b
; CHECK-NEXT:    [[R:%.*]] = mul i8 [[M1]], [[M1]]
; CHECK-NEXT:    ret i8 [[R]]
;
  %cmp1 = icmp slt i8 %a, %b
  %cmp2 = icmp slt i8 %b, %a
  %m1 = select i1 %cmp1, i8 %a, i8 %b
  %m2 = select i1 %cmp2, i8 %b, i8 %a
  %r = mul i8 %m1, %m2
  ret i8 %r
}

; Min/max can also have a swapped predicate and select operands.

define i1 @smin_swapped(i8 %a, i8 %b) {
; CHECK-LABEL: @smin_swapped(
; CHECK-NEXT:    [[CMP1:%.*]] = icmp sgt i8 %a, %b
; CHECK-NEXT:    [[CMP2:%.*]] = icmp slt i8 %a, %b
; CHECK-NEXT:    [[M1:%.*]] = select i1 [[CMP1]], i8 %b, i8 %a
; CHECK-NEXT:    ret i1 true
;
  %cmp1 = icmp sgt i8 %a, %b
  %cmp2 = icmp slt i8 %a, %b
  %m1 = select i1 %cmp1, i8 %b, i8 %a
  %m2 = select i1 %cmp2, i8 %a, i8 %b
  %r = icmp eq i8 %m2, %m1
  ret i1 %r
}

define i8 @smax_commute(i8 %a, i8 %b) {
; CHECK-LABEL: @smax_commute(
; CHECK-NEXT:    [[CMP1:%.*]] = icmp sgt i8 %a, %b
; CHECK-NEXT:    [[CMP2:%.*]] = icmp sgt i8 %b, %a
; CHECK-NEXT:    [[M1:%.*]] = select i1 [[CMP1]], i8 %a, i8 %b
; CHECK-NEXT:    ret i8 0
;
  %cmp1 = icmp sgt i8 %a, %b
  %cmp2 = icmp sgt i8 %b, %a
  %m1 = select i1 %cmp1, i8 %a, i8 %b
  %m2 = select i1 %cmp2, i8 %b, i8 %a
  %r = urem i8 %m2, %m1
  ret i8 %r
}

define i8 @smax_swapped(i8 %a, i8 %b) {
; CHECK-LABEL: @smax_swapped(
; CHECK-NEXT:    [[CMP1:%.*]] = icmp slt i8 %a, %b
; CHECK-NEXT:    [[CMP2:%.*]] = icmp sgt i8 %a, %b
; CHECK-NEXT:    [[M1:%.*]] = select i1 [[CMP1]], i8 %b, i8 %a
; CHECK-NEXT:    ret i8 1
;
  %cmp1 = icmp slt i8 %a, %b
  %cmp2 = icmp sgt i8 %a, %b
  %m1 = select i1 %cmp1, i8 %b, i8 %a
  %m2 = select i1 %cmp2, i8 %a, i8 %b
  %r = sdiv i8 %m1, %m2
  ret i8 %r
}

define i8 @umin_commute(i8 %a, i8 %b) {
; CHECK-LABEL: @umin_commute(
; CHECK-NEXT:    [[CMP1:%.*]] = icmp ult i8 %a, %b
; CHECK-NEXT:    [[CMP2:%.*]] = icmp ult i8 %b, %a
; CHECK-NEXT:    [[M1:%.*]] = select i1 [[CMP1]], i8 %a, i8 %b
; CHECK-NEXT:    ret i8 0
;
  %cmp1 = icmp ult i8 %a, %b
  %cmp2 = icmp ult i8 %b, %a
  %m1 = select i1 %cmp1, i8 %a, i8 %b
  %m2 = select i1 %cmp2, i8 %b, i8 %a
  %r = sub i8 %m2, %m1
  ret i8 %r
}

; Choose a vector type just to show that works.

define <2 x i8> @umin_swapped(<2 x i8> %a, <2 x i8> %b) {
; CHECK-LABEL: @umin_swapped(
; CHECK-NEXT:    [[CMP1:%.*]] = icmp ugt <2 x i8> %a, %b
; CHECK-NEXT:    [[CMP2:%.*]] = icmp ult <2 x i8> %a, %b
; CHECK-NEXT:    [[M1:%.*]] = select <2 x i1> [[CMP1]], <2 x i8> %b, <2 x i8> %a
; CHECK-NEXT:    ret <2 x i8> zeroinitializer
;
  %cmp1 = icmp ugt <2 x i8> %a, %b
  %cmp2 = icmp ult <2 x i8> %a, %b
  %m1 = select <2 x i1> %cmp1, <2 x i8> %b, <2 x i8> %a
  %m2 = select <2 x i1> %cmp2, <2 x i8> %a, <2 x i8> %b
  %r = sub <2 x i8> %m2, %m1
  ret <2 x i8> %r
}

define i8 @umax_commute(i8 %a, i8 %b) {
; CHECK-LABEL: @umax_commute(
; CHECK-NEXT:    [[CMP1:%.*]] = icmp ugt i8 %a, %b
; CHECK-NEXT:    [[CMP2:%.*]] = icmp ugt i8 %b, %a
; CHECK-NEXT:    [[M1:%.*]] = select i1 [[CMP1]], i8 %a, i8 %b
; CHECK-NEXT:    ret i8 1
;
  %cmp1 = icmp ugt i8 %a, %b
  %cmp2 = icmp ugt i8 %b, %a
  %m1 = select i1 %cmp1, i8 %a, i8 %b
  %m2 = select i1 %cmp2, i8 %b, i8 %a
  %r = udiv i8 %m1, %m2
  ret i8 %r
}

define i8 @umax_swapped(i8 %a, i8 %b) {
; CHECK-LABEL: @umax_swapped(
; CHECK-NEXT:    [[CMP1:%.*]] = icmp ult i8 %a, %b
; CHECK-NEXT:    [[CMP2:%.*]] = icmp ugt i8 %a, %b
; CHECK-NEXT:    [[M1:%.*]] = select i1 [[CMP1]], i8 %b, i8 %a
; CHECK-NEXT:    [[R:%.*]] = add i8 [[M1]], [[M1]]
; CHECK-NEXT:    ret i8 [[R]]
;
  %cmp1 = icmp ult i8 %a, %b
  %cmp2 = icmp ugt i8 %a, %b
  %m1 = select i1 %cmp1, i8 %b, i8 %a
  %m2 = select i1 %cmp2, i8 %a, i8 %b
  %r = add i8 %m2, %m1
  ret i8 %r
}

; Min/max may exist with non-canonical operands. Value tracking can match those.

define i8 @smax_nsw(i8 %a, i8 %b) {
; CHECK-LABEL: @smax_nsw(
; CHECK-NEXT:    [[SUB:%.*]] = sub nsw i8 %a, %b
; CHECK-NEXT:    [[CMP1:%.*]] = icmp slt i8 %a, %b
; CHECK-NEXT:    [[CMP2:%.*]] = icmp sgt i8 [[SUB]], 0
; CHECK-NEXT:    [[M1:%.*]] = select i1 [[CMP1]], i8 0, i8 [[SUB]]
; CHECK-NEXT:    ret i8 0
;
  %sub = sub nsw i8 %a, %b
  %cmp1 = icmp slt i8 %a, %b
  %cmp2 = icmp sgt i8 %sub, 0
  %m1 = select i1 %cmp1, i8 0, i8 %sub
  %m2 = select i1 %cmp2, i8 %sub, i8 0
  %r = sub i8 %m2, %m1
  ret i8 %r
}

