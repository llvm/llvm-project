target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

source_filename = "-"

%struct.A = type { i32 (...)** }
%struct.B = type { %struct.A }

@_ZTV1B = internal constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%struct.B*, i32)* @_ZN1B1fEi to i8*), i8* bitcast (i32 (%struct.A*, i32)* @_ZN1A1nEi to i8*)] }, !type !0, !type !1

define internal i32 @_ZN1B1fEi(%struct.B* %this, i32 %a) #0 {
   ret i32 0;
}

define internal i32 @_ZN1A1nEi(%struct.A* %this, i32 %a) #0 {
   ret i32 0;
}

define i32 @test2(%struct.B* %obj, i32 %a) {
entry:
  %0 = bitcast %struct.B* %obj to i8***
  %vtable2 = load i8**, i8*** %0
  %1 = bitcast i8** %vtable2 to i8*
  %p2 = call i1 @llvm.type.test(i8* %1, metadata !"_ZTS1B")
  call void @llvm.assume(i1 %p2)

  %fptrptr = getelementptr i8*, i8** %vtable2, i32 1
  %2 = bitcast i8** %fptrptr to i32 (%struct.B*, i32)**
  %fptr33 = load i32 (%struct.B*, i32)*, i32 (%struct.B*, i32)** %2, align 8

  %call4 = tail call i32 %fptr33(%struct.B* nonnull %obj, i32 %a)
  ret i32 %call4
}

attributes #0 = { noinline optnone }

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
