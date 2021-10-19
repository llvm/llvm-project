target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%aaa = type <{ %aab, i32, [4 x i8] }>
%aab = type { i64 }
%fum = type { %aac, i8, [7 x i8] }
%aac = type { [8 x i8] }

declare void @bar01(%aaa*)
declare void @bar02(%fum*)
