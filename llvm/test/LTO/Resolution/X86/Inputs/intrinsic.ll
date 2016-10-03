target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
%foo = type {  }
declare <4 x %foo*> @llvm.masked.load.v4p0foo.p0v4p0foo(<4 x %foo*>*, i32, <4 x i1>, <4 x %foo*>)
