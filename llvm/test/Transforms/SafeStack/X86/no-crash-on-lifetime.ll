; Check that the pass does not crash on the code.
; RUN: opt -safe-stack -S -mtriple=x86_64-pc-linux-gnu %s -o /dev/null

%class.F = type { %class.o, i8, [7 x i8] }
%class.o = type <{ ptr, i32, [4 x i8] }>

define dso_local void @_ZN1s1tE1F(ptr byval(%class.F) %g) local_unnamed_addr safestack align 32 {
entry:
  %ref.tmp.i.i.i = alloca i64, align 1
  call void undef(ptr %g)
  call void @llvm.lifetime.start.p0(i64 3, ptr %ref.tmp.i.i.i)
  ret void
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

