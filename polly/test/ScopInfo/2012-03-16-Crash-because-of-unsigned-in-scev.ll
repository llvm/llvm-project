; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-a0:0-n32"

@array = external global [64 x i8], align 8

define void @foo(ptr %A) nounwind {
entry:
  br label %if.then132

if.then132:
  %loaded = load i32, ptr %A
  %0 = icmp ugt i32 %loaded, 10
  %umax = select i1 %0, i32 %loaded, i32 10
  br label %do.body

do.body:
  %indvar = phi i32 [ %3, %do.body ], [ 0, %if.then132 ]
  %1 = add i32 0, %umax
  %2 = sub i32 %1, %indvar
  %arrayidx = getelementptr [64 x i8], ptr @array, i32 0, i32 %2
  store i8 1, ptr %arrayidx, align 1
  %3 = add i32 %indvar, 1
  %exitcond = icmp eq i32 %3, 20
  br i1 %exitcond, label %for.end, label %do.body

for.end:
  ret void
}

;CHECK: p0: (10 umax %loaded)
