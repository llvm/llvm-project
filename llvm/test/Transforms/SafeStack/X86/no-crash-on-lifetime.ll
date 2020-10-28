; Check that the pass does not crash on the code.
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu %s -o /dev/null

%class.F = type { %class.o, i8, [7 x i8] }
%class.o = type <{ i8*, i32, [4 x i8] }>

define dso_local void @_ZN1s1tE1F(%class.F* byval(%class.F) %g) local_unnamed_addr safestack align 32 {
entry:
  %ref.tmp.i.i.i = alloca i64, align 1
  call void undef(%class.F* %g)
  %ref.tmp.i.i.0..sroa_idx.i5 = bitcast i64* %ref.tmp.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 3, i8* %ref.tmp.i.i.0..sroa_idx.i5)
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

