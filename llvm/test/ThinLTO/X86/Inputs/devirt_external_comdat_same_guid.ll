target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

source_filename = "-"

%struct.A = type { ptr }
%struct.B = type { %struct.A }

$_ZTV1B = comdat any

@_ZTV1B = constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr undef, ptr @_ZN1B1fEi, ptr @_ZN1A1nEi] }, comdat, !type !0, !type !1

define i32 @_ZN1B1fEi(ptr %this, i32 %a) #0 comdat($_ZTV1B) {
   ret i32 0;
}

define i32 @_ZN1A1nEi(ptr %this, i32 %a) #0 comdat($_ZTV1B) {
   ret i32 0;
}

define i32 @test2(ptr %obj, i32 %a) {
entry:
  %vtable2 = load ptr, ptr %obj
  %p2 = call i1 @llvm.type.test(ptr %vtable2, metadata !"_ZTS1B")
  call void @llvm.assume(i1 %p2)

  %fptrptr = getelementptr ptr, ptr %vtable2, i32 1
  %fptr33 = load ptr, ptr %fptrptr, align 8

  %call4 = tail call i32 %fptr33(ptr nonnull %obj, i32 %a)
  ret i32 %call4
}

attributes #0 = { noinline optnone }

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
