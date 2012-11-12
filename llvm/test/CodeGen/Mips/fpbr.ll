; RUN: llc  < %s -march=mipsel | FileCheck %s

define void @func0(float %f2, float %f3) nounwind {
entry:
; CHECK: c.eq.s
; CHECK: bc1f
  %cmp = fcmp oeq float %f2, %f3
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  tail call void (...)* @g0() nounwind
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void (...)* @g1() nounwind
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

declare void @g0(...)

declare void @g1(...)

define void @func1(float %f2, float %f3) nounwind {
entry:
; CHECK: c.olt.s
; CHECK: bc1f
  %cmp = fcmp olt float %f2, %f3
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  tail call void (...)* @g0() nounwind
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void (...)* @g1() nounwind
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

define void @func2(float %f2, float %f3) nounwind {
entry:
; CHECK: c.ole.s
; CHECK: bc1t
  %cmp = fcmp ugt float %f2, %f3
  br i1 %cmp, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  tail call void (...)* @g0() nounwind
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void (...)* @g1() nounwind
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

define void @func3(double %f2, double %f3) nounwind {
entry:
; CHECK: c.eq.d
; CHECK: bc1f
  %cmp = fcmp oeq double %f2, %f3
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  tail call void (...)* @g0() nounwind
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void (...)* @g1() nounwind
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

define void @func4(double %f2, double %f3) nounwind {
entry:
; CHECK: c.olt.d
; CHECK: bc1f
  %cmp = fcmp olt double %f2, %f3
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  tail call void (...)* @g0() nounwind
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void (...)* @g1() nounwind
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

define void @func5(double %f2, double %f3) nounwind {
entry:
; CHECK: c.ole.d
; CHECK: bc1t
  %cmp = fcmp ugt double %f2, %f3
  br i1 %cmp, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  tail call void (...)* @g0() nounwind
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void (...)* @g1() nounwind
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}
