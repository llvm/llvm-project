target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.D = type { i32 (...)** }

@some_name = constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%struct.D*, i32)* @_ZN1D1mEi to i8*)] }, !type !3
@_ZTV1D = alias { [3 x i8*] }, { [3 x i8*] }* @some_name

define i32 @_ZN1D1mEi(%struct.D* %this, i32 %a) #0 {
   ret i32 0;
}

attributes #0 = { noinline optnone }

!3 = !{i64 16, !"_ZTS1D"}
