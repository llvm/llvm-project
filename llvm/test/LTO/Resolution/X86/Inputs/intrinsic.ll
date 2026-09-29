target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
%foo = type {  }
declare <4 x ptr> @llvm.masked.load.v4p0.p0(ptr, i32, <4 x i1>, <4 x ptr>)
