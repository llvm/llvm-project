; RUN: opt < %s -mtriple=systemz-unknown -mcpu=z15 -passes='cgscc(inline)' -disable-output \
; RUN:   -debug-only=inline,systemztti 2>&1 | FileCheck %s
; REQUIRES: asserts

; Check that the inlining threshold is incremented for a function using an
; argument only as a memcpy source.
;
; CHECK: Inlining calls in: root_function
; CHECK:     Inlining {{.*}} Call:   call void @leaf_function_A(ptr %Dst)
; CHECK:     ++ SZTTI Adding inlining bonus: 1000
; CHECK:     Inlining {{.*}} Call:   call void @leaf_function_B(ptr %Dst, ptr %Src)

define void @leaf_function_A(ptr %Dst)  {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %Dst, ptr undef, i64 16, i1 false)
  ret void
}

define void @leaf_function_B(ptr %Dst, ptr %Src)  {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %Dst, ptr %Src, i64 16, i1 false)
  ret void
}

define void @root_function(ptr %Dst, ptr %Src) {
entry:
  call void @leaf_function_A(ptr %Dst)
  call void @leaf_function_B(ptr %Dst, ptr %Src)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)

; Check that the inlining threshold is incremented in case of multiple
; accesses of a global variable by both caller and callee (which is true here
; after the first call is inlined).
;
; CHECK: Inlining calls in: Caller1
; CHECK: ++ SZTTI Adding inlining bonus: 1000

@GlobV = external global i32

define i64 @Caller1(i1 %cond1, i32 %0) #0 {
entry:
  br i1 %cond1, label %sw.bb3437, label %fake_end

common.ret:                                       ; preds = %fake_end, %sw.bb3437
  ret i64 0

sw.bb3437:                                        ; preds = %entry
  %call34652 = call i32 @Callee1(ptr null, i32 %0)
  br label %common.ret

fake_end:                                         ; preds = %entry
  %call57981 = call i32 @Callee1(ptr null, i32 0)
  br label %common.ret
}

define i32 @Callee1(ptr %rex, i32 %parenfloor) #0 {
entry:
  %cmp21 = icmp slt i32 %parenfloor, 0
  br i1 %cmp21, label %for.body, label %for.end

common.ret:                                       ; preds = %for.end, %for.body
  ret i32 0

for.body:                                         ; preds = %entry
  %0 = load i32, ptr @GlobV, align 4
  %inc = or i32 %0, 1
  store i32 %inc, ptr @GlobV, align 4
  store i64 0, ptr %rex, align 8
  %1 = load i32, ptr @GlobV, align 4
  %inc28 = or i32 %1, 1
  store i32 %inc28, ptr @GlobV, align 4
  store i64 0, ptr %rex, align 8
  %2 = load i32, ptr @GlobV, align 4
  %inc35 = or i32 %2, 1
  store i32 %inc35, ptr @GlobV, align 4
  store i32 0, ptr %rex, align 8
  br label %common.ret

for.end:                                          ; preds = %entry
  store i32 0, ptr @GlobV, align 4
  store i32 0, ptr %rex, align 8
  %3 = load i32, ptr @GlobV, align 4
  %inc42 = or i32 %3, 1
  store i32 %inc42, ptr @GlobV, align 4
  store i32 0, ptr %rex, align 8
  %4 = load i32, ptr @GlobV, align 4
  %inc48 = or i32 %4, 1
  store i32 %inc48, ptr @GlobV, align 4
  br label %common.ret
}

; Check that the inlining threshold is incremented for a function that is
; accessing an alloca of the caller multiple times.
;
; CHECK: Inlining calls in: Caller2
; CHECK: ++ SZTTI Adding inlining bonus: 550

define i1 @Caller2() {
entry:
  %A = alloca [80 x i64], align 8
  call void @Callee2(ptr %A)
  ret i1 false
}

define void @Callee2(ptr nocapture readonly %Arg) {
entry:
  %nonzero = getelementptr i8, ptr %Arg, i64 48
  %0 = load i32, ptr %nonzero, align 8
  %tobool1.not = icmp eq i32 %0, 0
  br i1 %tobool1.not, label %if.else38, label %if.then2

if.then2:                                         ; preds = %entry
  %1 = load i32, ptr %Arg, align 4
  %tobool4.not = icmp eq i32 %1, 0
  br i1 %tobool4.not, label %common.ret, label %if.then5

if.then5:                                         ; preds = %if.then2
  %2 = load double, ptr %Arg, align 8
  %slab_den = getelementptr i8, ptr %Arg, i64 24
  %3 = load double, ptr %slab_den, align 8
  %mul = fmul double %2, %3
  %cmp = fcmp olt double %mul, 0.000000e+00
  br i1 %cmp, label %common.ret, label %if.end55

common.ret:                                       ; preds = %if.end100, %if.else79, %if.end55, %if.else38, %if.then5, %if.then2
  ret void

if.else38:                                        ; preds = %entry
  %4 = load double, ptr %Arg, align 8
  %cmp52 = fcmp ogt double %4, 0.000000e+00
  br i1 %cmp52, label %common.ret, label %if.end55

if.end55:                                         ; preds = %if.else38, %if.then5
  %arrayidx57 = getelementptr i8, ptr %Arg, i64 52
  %5 = load i32, ptr %arrayidx57, align 4
  %tobool58.not = icmp eq i32 %5, 0
  br i1 %tobool58.not, label %common.ret, label %if.then59

if.then59:                                        ; preds = %if.end55
  %arrayidx61 = getelementptr i8, ptr %Arg, i64 64
  %6 = load i32, ptr %arrayidx61, align 4
  %tobool62.not = icmp eq i32 %6, 0
  br i1 %tobool62.not, label %if.else79, label %if.end100

if.else79:                                        ; preds = %if.then59
  %arrayidx84 = getelementptr i8, ptr %Arg, i64 8
  %7 = load double, ptr %arrayidx84, align 8
  %arrayidx87 = getelementptr i8, ptr %Arg, i64 32
  %8 = load double, ptr %arrayidx87, align 8
  %mul88 = fmul double %7, %8
  %9 = fcmp olt double %mul88, 0.000000e+00
  br i1 %9, label %common.ret, label %if.end100

if.end100:                                        ; preds = %if.else79, %if.then59
  %arrayidx151 = getelementptr i8, ptr %Arg, i64 16
  %10 = load double, ptr %arrayidx151, align 8
  %arrayidx154 = getelementptr i8, ptr %Arg, i64 40
  %11 = load double, ptr %arrayidx154, align 8
  %mul155 = fmul double %10, %11
  %cmp181 = fcmp olt double %mul155, 0.000000e+00
  br label %common.ret
}
