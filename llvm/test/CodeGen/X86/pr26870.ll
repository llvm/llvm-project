; RUN: llc < %s -mtriple=i686-pc-windows-msvc18.0.0 -mcpu=pentium4

define x86_thiscallcc ptr @fn4(ptr %this, ptr dereferenceable(1) %p1) {
entry:
  %call.i = tail call x86_thiscallcc i64 @fn1(ptr %this)
  %0 = load i32, ptr %this, align 4
  %call.i8 = tail call x86_thiscallcc i64 @fn1(ptr %this)
  %1 = insertelement <2 x i64> undef, i64 %call.i, i32 0
  %2 = insertelement <2 x i64> %1, i64 %call.i8, i32 1
  %3 = add nsw <2 x i64> %2, <i64 7, i64 7>
  %4 = sdiv <2 x i64> %3, <i64 8, i64 8>
  %5 = add nsw <2 x i64> %4, <i64 1, i64 1>
  %6 = load i32, ptr %this, align 4
  %7 = insertelement <2 x i32> undef, i32 %0, i32 0
  %8 = insertelement <2 x i32> %7, i32 %6, i32 1
  %9 = zext <2 x i32> %8 to <2 x i64>
  %10 = srem <2 x i64> %5, %9
  %11 = sub <2 x i64> %5, %10
  %12 = trunc <2 x i64> %11 to <2 x i32>
  %13 = extractelement <2 x i32> %12, i32 0
  %14 = extractelement <2 x i32> %12, i32 1
  %cmp = icmp eq i32 %13, %14
  br i1 %cmp, label %if.then, label %cleanup

if.then:
  %call4 = tail call x86_thiscallcc ptr @fn3(ptr nonnull %p1)
  br label %cleanup

cleanup:
  %retval.0 = phi ptr [ %call4, %if.then ], [ undef, %entry ]
  ret ptr %retval.0
}

declare x86_thiscallcc ptr @fn3(ptr)
declare x86_thiscallcc i64 @fn1(ptr)
