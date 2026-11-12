target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.B2 = type { %struct.A2 }
%struct.A2 = type { ptr }
%struct.B3 = type { %struct.A3 }
%struct.A3 = type { ptr }

@_ZTV1B2 = constant { [3 x ptr] } { [3 x ptr] [ptr undef, ptr undef, ptr undef] }, !type !0

@_ZTV1B3 = constant { [3 x ptr] } { [3 x ptr] [ptr undef, ptr undef, ptr undef] }, !type !1

define void @test2(ptr %b) {
entry:
  %vtable2 = load ptr, ptr %b
  %0 = tail call i1 @llvm.type.test(ptr %vtable2, metadata !"_ZTS1A2")
  br i1 %0, label %cont, label %trap

trap:
  tail call void @llvm.trap()
  unreachable

cont:
  ret void
}

define void @test1(ptr %b) {
entry:
  %vtable2 = load ptr, ptr %b
  %0 = tail call i1 @llvm.type.test(ptr %vtable2, metadata !"_ZTS1A3")
  br i1 %0, label %cont, label %trap

trap:
  tail call void @llvm.trap()
  unreachable

cont:
  ret void
}

@test3 = hidden unnamed_addr alias void (ptr), ptr @test1

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.trap()

!0 = !{i64 16, !"_ZTS1A2"}
!1 = !{i64 16, !"_ZTS1A3"}
