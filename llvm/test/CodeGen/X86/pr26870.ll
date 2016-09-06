; RUN: llc < %s -mtriple=i686-pc-windows-msvc18.0.0 -mcpu=pentium4

define x86_thiscallcc i32* @fn4(i32* %this, i8* dereferenceable(1) %p1) {
entry:
  %DL = getelementptr inbounds i32, i32* %this, i32 0
  %call.i = tail call x86_thiscallcc i64 @fn1(i32* %DL)
  %getTypeAllocSize___trans_tmp_2.i = getelementptr inbounds i32, i32* %this, i32 0
  %0 = load i32, i32* %getTypeAllocSize___trans_tmp_2.i, align 4
  %call.i8 = tail call x86_thiscallcc i64 @fn1(i32* %DL)
  %1 = insertelement <2 x i64> undef, i64 %call.i, i32 0
  %2 = insertelement <2 x i64> %1, i64 %call.i8, i32 1
  %3 = add nsw <2 x i64> %2, <i64 7, i64 7>
  %4 = sdiv <2 x i64> %3, <i64 8, i64 8>
  %5 = add nsw <2 x i64> %4, <i64 1, i64 1>
  %6 = load i32, i32* %getTypeAllocSize___trans_tmp_2.i, align 4
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
  %call4 = tail call x86_thiscallcc i32* @fn3(i8* nonnull %p1)
  br label %cleanup

cleanup:
  %retval.0 = phi i32* [ %call4, %if.then ], [ undef, %entry ]
  ret i32* %retval.0
}

declare x86_thiscallcc i32* @fn3(i8*)
declare x86_thiscallcc i64 @fn1(i32*)
