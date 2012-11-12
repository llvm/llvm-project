; RUN: opt -instcombine -S < %s | FileCheck %s
; Radar 10803727
@.str = private unnamed_addr constant [35 x i8] c"\0Ain_range input (should be 0): %f\0A\00", align 1
@.str1 = external hidden unnamed_addr constant [35 x i8], align 1

declare i32 @printf(i8*, ...)
define i64 @_Z8tempCastj(i32 %val) uwtable ssp {
entry:
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([35 x i8]* @.str1, i64 0, i64 0), i32 %val)
  %conv = uitofp i32 %val to double
  %call.i = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([35 x i8]* @.str, i64 0, i64 0), double %conv)
  %cmp.i = fcmp oge double %conv, -1.000000e+00
  br i1 %cmp.i, label %land.rhs.i, label %if.end.critedge
; CHECK:  br i1 true, label %land.rhs.i, label %if.end.critedge

land.rhs.i:                                       ; preds = %entry
  %cmp1.i = fcmp olt double %conv, 1.000000e+00
  br i1 %cmp1.i, label %if.then, label %if.end

if.then:                                          ; preds = %land.rhs.i
  %add = fadd double %conv, 5.000000e-01
  %conv3 = fptosi double %add to i64
  br label %return

if.end.critedge:                                  ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %if.end.critedge, %land.rhs.i
  br label %return

return:                                           ; preds = %if.end, %if.then
  %retval.0 = phi i64 [ %conv3, %if.then ], [ -1, %if.end ]
  ret i64 %retval.0
}

