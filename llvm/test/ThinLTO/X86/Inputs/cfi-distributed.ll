target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.B2 = type { %struct.A2 }
%struct.A2 = type { i32 (...)** }

@_ZTV1B2 = constant { [3 x i8*] } { [3 x i8*] [i8* undef, i8* undef, i8* undef] }, !type !0

define void @test2(i8* %b) {
entry:
  %0 = bitcast i8* %b to i8**
  %vtable2 = load i8*, i8** %0
  %1 = tail call i1 @llvm.type.test(i8* %vtable2, metadata !"_ZTS1A2")
  br i1 %1, label %cont, label %trap

trap:
  tail call void @llvm.trap()
  unreachable

cont:
  ret void
}

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.trap()

!0 = !{i64 16, !"_ZTS1A2"}
!1 = !{i64 16, !"_ZTS1B2"}
