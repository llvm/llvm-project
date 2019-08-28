target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.B2 = type { %struct.A2 }
%struct.A2 = type { i32 (...)** }
%struct.B3 = type { %struct.A3 }
%struct.A3 = type { i32 (...)** }

@_ZTV1B2 = constant { [3 x i8*] } { [3 x i8*] [i8* undef, i8* undef, i8* undef] }, !type !0

@_ZTV1B3 = constant { [3 x i8*] } { [3 x i8*] [i8* undef, i8* undef, i8* undef] }, !type !1

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

define void @test1(i8* %b) {
entry:
  %0 = bitcast i8* %b to i8**
  %vtable2 = load i8*, i8** %0
  %1 = tail call i1 @llvm.type.test(i8* %vtable2, metadata !"_ZTS1A3")
  br i1 %1, label %cont, label %trap

trap:
  tail call void @llvm.trap()
  unreachable

cont:
  ret void
}

@test3 = hidden unnamed_addr alias void (i8*), void (i8*)* @test1

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.trap()

!0 = !{i64 16, !"_ZTS1A2"}
!1 = !{i64 16, !"_ZTS1A3"}
