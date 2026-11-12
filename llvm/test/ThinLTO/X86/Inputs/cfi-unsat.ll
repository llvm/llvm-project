target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { ptr }
%struct.B = type { ptr }

@_ZTV1B = linkonce_odr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr undef, ptr @_ZN1B1fEi, ptr @_ZN1B1nEi] }, !type !0

$test = comdat any
$testb = comdat any

define linkonce_odr i32 @test(ptr %obj, i32 %a) comdat {
entry:
  %vtable5 = load ptr, ptr %obj

  %0 = tail call { ptr, i1 } @llvm.type.checked.load(ptr %vtable5, i32 8, metadata !"_ZTS1A")
  %1 = extractvalue { ptr, i1 } %0, 1
  br i1 %1, label %cont, label %trap

trap:
  tail call void @llvm.trap()
  unreachable

cont:
  %2 = extractvalue { ptr, i1 } %0, 0

  %call = tail call i32 %2(ptr nonnull %obj, i32 %a)

  ret i32 %call
}

define linkonce_odr i32 @testb(ptr %obj, i32 %a) comdat {
entry:
  %vtable5 = load ptr, ptr %obj

  %0 = tail call { ptr, i1 } @llvm.type.checked.load(ptr %vtable5, i32 0, metadata !"_ZTS1A")
  %1 = extractvalue { ptr, i1 } %0, 1
  br i1 %1, label %cont, label %trap

trap:
  tail call void @llvm.trap()
  unreachable

cont:
  %2 = extractvalue { ptr, i1 } %0, 0

  %call = tail call i32 %2(ptr nonnull %obj, i32 %a)

  ret i32 %call
}

declare { ptr, i1 } @llvm.type.checked.load(ptr, i32, metadata)
declare void @llvm.trap()

define internal i32 @_ZN1B1fEi(ptr %this, i32 %a) {
entry:
   ret i32 0
}
define internal i32 @_ZN1B1nEi(ptr %this, i32 %a) {
entry:
   ret i32 0
}

!0 = !{i64 16, !"_ZTS1B"}
