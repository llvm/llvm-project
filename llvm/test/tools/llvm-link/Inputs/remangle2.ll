target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%fum = type { %aab, i8, [7 x i8] }
%aab = type { %aba }
%aba = type { [8 x i8] }
%fum.1 = type { %abb, i8, [7 x i8] }
%abb = type { %abc }
%abc = type { [4 x i8] }

declare void @foo(%fum*)

declare %fum.1** @"llvm.ssa.copy.p0p0s_fum.1s"(%fum.1**)

declare %fum** @"llvm.ssa.copy.p0p0s_fums"(%fum**)

define void @foo1(%fum** %a, %fum.1 ** %b) {
  %b.copy = call %fum.1** @"llvm.ssa.copy.p0p0s_fum.1s"(%fum.1** %b)
  %a.copy = call %fum** @"llvm.ssa.copy.p0p0s_fums"(%fum** %a)
  ret void
}

define void @foo2(%fum.1 ** %b, %fum** %a) {
  %a.copy = call %fum** @"llvm.ssa.copy.p0p0s_fums"(%fum** %a)
  %b.copy = call %fum.1** @"llvm.ssa.copy.p0p0s_fum.1s"(%fum.1** %b)
  ret void
}
