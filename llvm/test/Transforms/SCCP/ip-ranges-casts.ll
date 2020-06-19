; RUN: opt < %s -ipsccp -S | FileCheck %s

; x = [100, 301)
define internal i1 @f.trunc(i32 %x) {
; CHECK-LABEL: define internal i1 @f.trunc(i32 %x) {
; CHECK-NEXT:    %t.1 = trunc i32 %x to i16
; CHECK-NEXT:    %c.2 = icmp sgt i16 %t.1, 299
; CHECK-NEXT:    %c.4 = icmp slt i16 %t.1, 101
; CHECK-NEXT:    %res.1 = add i1 false, %c.2
; CHECK-NEXT:    %res.2 = add i1 %res.1, false
; CHECK-NEXT:    %res.3 = add i1 %res.2, %c.4
; CHECK-NEXT:    %t.2 = trunc i32 %x to i8
; CHECK-NEXT:    %c.5 = icmp sgt i8 %t.2, 44
; CHECK-NEXT:    %c.6 = icmp sgt i8 %t.2, 43
; CHECK-NEXT:    %c.7 = icmp slt i8 %t.2, 100
; CHECK-NEXT:    %c.8 = icmp slt i8 %t.2, 101
; CHECK-NEXT:    %res.4 = add i1 %res.3, %c.5
; CHECK-NEXT:    %res.5 = add i1 %res.4, %c.6
; CHECK-NEXT:    %res.6 = add i1 %res.5, %c.7
; CHECK-NEXT:    %res.7 = add i1 %res.6, %c.8
; CHECK-NEXT:    ret i1 %res.7

  %t.1 = trunc i32 %x to i16
  %c.1 = icmp sgt i16 %t.1, 300
  %c.2 = icmp sgt i16 %t.1, 299
  %c.3 = icmp slt i16 %t.1, 100
  %c.4 = icmp slt i16 %t.1, 101
  %res.1 = add i1 %c.1, %c.2
  %res.2 = add i1 %res.1, %c.3
  %res.3 = add i1 %res.2, %c.4
  %t.2 = trunc i32 %x to i8
  %c.5 = icmp sgt i8 %t.2, 300
  %c.6 = icmp sgt i8 %t.2, 299
  %c.7 = icmp slt i8 %t.2, 100
  %c.8 = icmp slt i8 %t.2, 101
  %res.4 = add i1 %res.3, %c.5
  %res.5 = add i1 %res.4, %c.6
  %res.6 = add i1 %res.5, %c.7
  %res.7 = add i1 %res.6, %c.8
  ret i1 %res.7
}

define i1 @caller1() {
; CHECK-LABEL:  define i1 @caller1() {
; CHECK-NEXT:    %call.1 = tail call i1 @f.trunc(i32 100)
; CHECK-NEXT:    %call.2 = tail call i1 @f.trunc(i32 300)
; CHECK-NEXT:    %res = and i1 %call.1, %call.2
; CHECK-NEXT:    ret i1 %res
;
  %call.1 = tail call i1 @f.trunc(i32 100)
  %call.2 = tail call i1 @f.trunc(i32 300)
  %res = and i1 %call.1, %call.2
  ret i1 %res
}


; x = [100, 301)
define internal i1 @f.zext(i32 %x, i32 %y) {
; CHECK-LABEL: define internal i1 @f.zext(i32 %x, i32 %y) {
; CHECK-NEXT:    %t.1 = zext i32 %x to i64
; CHECK-NEXT:    %c.2 = icmp sgt i64 %t.1, 299
; CHECK-NEXT:    %c.4 = icmp slt i64 %t.1, 101
; CHECK-NEXT:    %res.1 = add i1 false, %c.2
; CHECK-NEXT:    %res.2 = add i1 %res.1, false
; CHECK-NEXT:    %res.3 = add i1 %res.2, %c.4
; CHECK-NEXT:    %t.2 = zext i32 %y to i64
; CHECK-NEXT:    %c.5 = icmp sgt i64 %t.2, 300
; CHECK-NEXT:    %c.6 = icmp sgt i64 %t.2, 299
; CHECK-NEXT:    %c.8 = icmp slt i64 %t.2, 1
; CHECK-NEXT:    %res.4 = add i1 %res.3, %c.5
; CHECK-NEXT:    %res.5 = add i1 %res.4, %c.6
; CHECK-NEXT:    %res.6 = add i1 %res.5, false
; CHECK-NEXT:    %res.7 = add i1 %res.6, %c.8
; CHECK-NEXT:    ret i1 %res.7

  %t.1 = zext i32 %x to i64
  %c.1 = icmp sgt i64 %t.1, 300
  %c.2 = icmp sgt i64 %t.1, 299
  %c.3 = icmp slt i64 %t.1, 100
  %c.4 = icmp slt i64 %t.1, 101
  %res.1 = add i1 %c.1, %c.2
  %res.2 = add i1 %res.1, %c.3
  %res.3 = add i1 %res.2, %c.4
  %t.2 = zext i32 %y to i64
  %c.5 = icmp sgt i64 %t.2, 300
  %c.6 = icmp sgt i64 %t.2, 299
  %c.7 = icmp slt i64 %t.2, 0
  %c.8 = icmp slt i64 %t.2, 1
  %res.4 = add i1 %res.3, %c.5
  %res.5 = add i1 %res.4, %c.6
  %res.6 = add i1 %res.5, %c.7
  %res.7 = add i1 %res.6, %c.8
  ret i1 %res.7
}

define i1 @caller.zext() {
; CHECK-LABEL:  define i1 @caller.zext() {
; CHECK-NEXT:    %call.1 = tail call i1 @f.zext(i32 100, i32 -120)
; CHECK-NEXT:    %call.2 = tail call i1 @f.zext(i32 300, i32 900)
; CHECK-NEXT:    %res = and i1 %call.1, %call.2
; CHECK-NEXT:    ret i1 %res
;
  %call.1 = tail call i1 @f.zext(i32 100, i32 -120)
  %call.2 = tail call i1 @f.zext(i32 300, i32 900)
  %res = and i1 %call.1, %call.2
  ret i1 %res
}

; x = [100, 301)
define internal i1 @f.sext(i32 %x, i32 %y) {
; CHECK-LABEL: define internal i1 @f.sext(i32 %x, i32 %y) {
; CHECK-NEXT:    [[T_1:%.*]] = zext i32 %x to i64
; CHECK-NEXT:    %c.2 = icmp sgt i64 [[T_1]], 299
; CHECK-NEXT:    %c.4 = icmp slt i64 [[T_1]], 101
; CHECK-NEXT:    %res.1 = add i1 false, %c.2
; CHECK-NEXT:    %res.2 = add i1 %res.1, false
; CHECK-NEXT:    %res.3 = add i1 %res.2, %c.4
; CHECK-NEXT:    %t.2 = sext i32 %y to i64
; CHECK-NEXT:    %c.6 = icmp sgt i64 %t.2, 899
; CHECK-NEXT:    %c.8 = icmp slt i64 %t.2, -119
; CHECK-NEXT:    %res.4 = add i1 %res.3, false
; CHECK-NEXT:    %res.5 = add i1 %res.4, %c.6
; CHECK-NEXT:    %res.6 = add i1 %res.5, false
; CHECK-NEXT:    %res.7 = add i1 %res.6, %c.8
; CHECK-NEXT:    ret i1 %res.7
;
  %t.1 = sext i32 %x to i64
  %c.1 = icmp sgt i64 %t.1, 300
  %c.2 = icmp sgt i64 %t.1, 299
  %c.3 = icmp slt i64 %t.1, 100
  %c.4 = icmp slt i64 %t.1, 101
  %res.1 = add i1 %c.1, %c.2
  %res.2 = add i1 %res.1, %c.3
  %res.3 = add i1 %res.2, %c.4
  %t.2 = sext i32 %y to i64
  %c.5 = icmp sgt i64 %t.2, 900
  %c.6 = icmp sgt i64 %t.2, 899
  %c.7 = icmp slt i64 %t.2, -120
  %c.8 = icmp slt i64 %t.2, -119
  %res.4 = add i1 %res.3, %c.5
  %res.5 = add i1 %res.4, %c.6
  %res.6 = add i1 %res.5, %c.7
  %res.7 = add i1 %res.6, %c.8
  ret i1 %res.7
}

define i1 @caller.sext() {
; CHECK-LABEL:  define i1 @caller.sext() {
; CHECK-NEXT:    %call.1 = tail call i1 @f.sext(i32 100, i32 -120)
; CHECK-NEXT:    %call.2 = tail call i1 @f.sext(i32 300, i32 900)
; CHECK-NEXT:    %res = and i1 %call.1, %call.2
; CHECK-NEXT:    ret i1 %res
;
  %call.1 = tail call i1 @f.sext(i32 100, i32 -120)
  %call.2 = tail call i1 @f.sext(i32 300, i32 900)
  %res = and i1 %call.1, %call.2
  ret i1 %res
}

; There's nothing we can do besides going to the full range or overdefined.
define internal i1 @f.fptosi(i32 %x) {
; CHECK-LABEL: define internal i1 @f.fptosi(i32 %x) {
; CHECK-NEXT:    %to.double = sitofp i32 %x to double
; CHECK-NEXT:    %add = fadd double 0.000000e+00, %to.double
; CHECK-NEXT:    %to.i32 = fptosi double %add to i32
; CHECK-NEXT:    %c.1 = icmp sgt i32 %to.i32, 300
; CHECK-NEXT:    %c.2 = icmp sgt i32 %to.i32, 299
; CHECK-NEXT:    %c.3 = icmp slt i32 %to.i32, 100
; CHECK-NEXT:    %c.4 = icmp slt i32 %to.i32, 101
; CHECK-NEXT:    %res.1 = add i1 %c.1, %c.2
; CHECK-NEXT:    %res.2 = add i1 %res.1, %c.3
; CHECK-NEXT:    %res.3 = add i1 %res.2, %c.4
; CHECK-NEXT:    ret i1 %res.3
;
  %to.double = sitofp i32 %x to double
  %add = fadd double 0.000000e+00, %to.double
  %to.i32 = fptosi double %add to i32
  %c.1 = icmp sgt i32 %to.i32, 300
  %c.2 = icmp sgt i32 %to.i32, 299
  %c.3 = icmp slt i32 %to.i32, 100
  %c.4 = icmp slt i32 %to.i32, 101
  %res.1 = add i1 %c.1, %c.2
  %res.2 = add i1 %res.1, %c.3
  %res.3 = add i1 %res.2, %c.4
  ret i1 %res.3
}

define i1 @caller.fptosi() {
; CHECK-LABEL:  define i1 @caller.fptosi() {
; CHECK-NEXT:    %call.1 = tail call i1 @f.fptosi(i32 100)
; CHECK-NEXT:    %call.2 = tail call i1 @f.fptosi(i32 300)
; CHECK-NEXT:    %res = and i1 %call.1, %call.2
; CHECK-NEXT:    ret i1 %res
;
  %call.1 = tail call i1 @f.fptosi(i32 100)
  %call.2 = tail call i1 @f.fptosi(i32 300)
  %res = and i1 %call.1, %call.2
  ret i1 %res
}

; There's nothing we can do besides going to the full range or overdefined.
define internal i1 @f.fpext(i16 %x) {
; CHECK-LABEL: define internal i1 @f.fpext(i16 %x) {
; CHECK-NEXT:    %to.float = sitofp i16 %x to float
; CHECK-NEXT:    %to.double = fpext float %to.float to double
; CHECK-NEXT:    %to.i64 = fptoui float %to.float to i64
; CHECK-NEXT:    %c.1 = icmp sgt i64 %to.i64, 300
; CHECK-NEXT:    %c.2 = icmp sgt i64 %to.i64, 299
; CHECK-NEXT:    %c.3 = icmp slt i64 %to.i64, 100
; CHECK-NEXT:    %c.4 = icmp slt i64 %to.i64, 101
; CHECK-NEXT:    %res.1 = add i1 %c.1, %c.2
; CHECK-NEXT:    %res.2 = add i1 %res.1, %c.3
; CHECK-NEXT:    %res.3 = add i1 %res.2, %c.4
; CHECK-NEXT:    ret i1 %res.3
;
  %to.float = sitofp i16 %x to float
  %to.double = fpext float %to.float  to double
  %to.i64= fptoui float %to.float to i64
  %c.1 = icmp sgt i64 %to.i64, 300
  %c.2 = icmp sgt i64 %to.i64, 299
  %c.3 = icmp slt i64 %to.i64, 100
  %c.4 = icmp slt i64 %to.i64, 101
  %res.1 = add i1 %c.1, %c.2
  %res.2 = add i1 %res.1, %c.3
  %res.3 = add i1 %res.2, %c.4
  ret i1 %res.3
}

; There's nothing we can do besides going to the full range or overdefined.
define i1 @caller.fpext() {
; CHECK-LABEL:  define i1 @caller.fpext() {
; CHECK-NEXT:    %call.1 = tail call i1 @f.fpext(i16 100)
; CHECK-NEXT:    %call.2 = tail call i1 @f.fpext(i16 300)
; CHECK-NEXT:    %res = and i1 %call.1, %call.2
; CHECK-NEXT:    ret i1 %res
;
  %call.1 = tail call i1 @f.fpext(i16 100)
  %call.2 = tail call i1 @f.fpext(i16 300)
  %res = and i1 %call.1, %call.2
  ret i1 %res
}

; There's nothing we can do besides going to the full range or overdefined.
define internal i1 @f.inttoptr.ptrtoint(i64 %x) {
; CHECK-LABEL: define internal i1 @f.inttoptr.ptrtoint(i64 %x) {
; CHECK-NEXT:    %to.ptr = inttoptr i64 %x to i8*
; CHECK-NEXT:    %to.i64 = ptrtoint i8* %to.ptr to i64
; CHECK-NEXT:    %c.1 = icmp sgt i64 %to.i64, 300
; CHECK-NEXT:    %c.2 = icmp sgt i64 %to.i64, 299
; CHECK-NEXT:    %c.3 = icmp slt i64 %to.i64, 100
; CHECK-NEXT:    %c.4 = icmp slt i64 %to.i64, 101
; CHECK-NEXT:    %res.1 = add i1 %c.1, %c.2
; CHECK-NEXT:    %res.2 = add i1 %res.1, %c.3
; CHECK-NEXT:    %res.3 = add i1 %res.2, %c.4
; CHECK-NEXT:    ret i1 %res.3
;
  %to.ptr = inttoptr i64 %x to i8*
  %to.i64 = ptrtoint i8* %to.ptr to i64
  %c.1 = icmp sgt i64 %to.i64, 300
  %c.2 = icmp sgt i64 %to.i64, 299
  %c.3 = icmp slt i64 %to.i64, 100
  %c.4 = icmp slt i64 %to.i64, 101
  %res.1 = add i1 %c.1, %c.2
  %res.2 = add i1 %res.1, %c.3
  %res.3 = add i1 %res.2, %c.4
  ret i1 %res.3
}

define i1 @caller.inttoptr.ptrtoint() {
; CHECK-LABEL:  define i1 @caller.inttoptr.ptrtoint() {
; CHECK-NEXT:    %call.1 = tail call i1 @f.inttoptr.ptrtoint(i64 100)
; CHECK-NEXT:    %call.2 = tail call i1 @f.inttoptr.ptrtoint(i64 300)
; CHECK-NEXT:    %res = and i1 %call.1, %call.2
; CHECK-NEXT:    ret i1 %res
;
  %call.1 = tail call i1 @f.inttoptr.ptrtoint(i64 100)
  %call.2 = tail call i1 @f.inttoptr.ptrtoint(i64 300)
  %res = and i1 %call.1, %call.2
  ret i1 %res
}

; Make sure we do not create constant ranges for int to fp casts.
define i1 @int_range_to_double_cast(i32 %a) {
; CHECK-LABEL: define i1 @int_range_to_double_cast(i32 %a)
; CHECK-NEXT:    %r = and i32 %a, 255
; CHECK-NEXT:    %tmp4 = sitofp i32 %r to double
; CHECK-NEXT:    %tmp10 = fadd double 0.000000e+00, %tmp4
; CHECK-NEXT:    %tmp11 = fcmp olt double %tmp4, %tmp10
; CHECK-NEXT:    ret i1 %tmp11
;
  %r = and i32 %a, 255
  %tmp4 = sitofp i32 %r to double
  %tmp10 = fadd double 0.000000e+00, %tmp4
  %tmp11 = fcmp olt double %tmp4, %tmp10
  ret i1 %tmp11
}

; Make sure we do not use ranges to propagate info from vectors.
define i16 @vector_binop_and_cast() {
; CHECK-LABEL: define i16 @vector_binop_and_cast(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %vecinit7 = insertelement <8 x i16> <i16 undef, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, i16 undef, i32 0
; CHECK-NEXT:    %rem = srem <8 x i16> <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>, %vecinit7
; CHECK-NEXT:    %0 = bitcast <8 x i16> %rem to i128
; CHECK-NEXT:    %1 = trunc i128 %0 to i16
; CHECK-NEXT:    ret i16 %1
entry:
  %vecinit7 = insertelement <8 x i16> <i16 undef, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, i16 undef, i32 0
  %rem = srem <8 x i16> <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>, %vecinit7
  %0 = bitcast <8 x i16> %rem to i128
  %1 = trunc i128 %0 to i16
  ret i16 %1
}
