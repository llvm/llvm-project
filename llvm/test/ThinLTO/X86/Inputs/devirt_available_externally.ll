target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.D = type { ptr }

@_ZTV1D = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr undef, ptr @_ZN1D1mEi] }, !type !3

define i32 @_ZN1D1mEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}

attributes #0 = { noinline optnone }

!3 = !{i64 16, !"_ZTS1D"}
