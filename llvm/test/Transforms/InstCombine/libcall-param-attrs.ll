; RUN: opt < %s -mtriple=s390x-linux-gnu -instcombine -S | FileCheck %s
;
; Check that instcombiner creates libcalls with parameter extensions per the
; prototype.

declare dso_local i8* @strchr(i8*, i32) local_unnamed_addr #1
declare dso_local i8* @memchr(i8*, i32 signext, i64)
declare void @llvm.assume(i1 noundef)
@0 = private unnamed_addr constant [21 x i8] c"000000000000000000000", align 2
define void @fun0(i32 %arg1) {
; CHECK: define void @fun0
; CHECK: call i8* @memchr{{.*}}, i32 signext %arg1, i64 22) #0
bb:
  %i = call i8* @strchr(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @0, i64 0, i64 0), i32 signext %arg1)
  %i3 = icmp ne i8* %i, null
  call void @llvm.assume(i1 %i3)
  ret void
}

declare dso_local double @pow(double, double)
define void @fun1(i32* %i5) {
; CHECK: define void @fun1
; CHECK: call double @ldexp{{.*}}, i32 signext %i19) #2
bb:
  %i19 = load i32, i32* %i5, align 4
  %i20 = sitofp i32 %i19 to double
  %i21 = call double @pow(double 2.000000e+00, double %i20)
  ret void
}
