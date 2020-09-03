target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { i32 (...)** }
%struct.B = type { i32 (...)** }

@_ZTV1B = linkonce_odr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%struct.B*, i32)* @_ZN1B1fEi to i8*), i8* bitcast (i32 (%struct.B*, i32)* @_ZN1B1nEi to i8*)] }, !type !0

$test = comdat any
$testb = comdat any

define linkonce_odr i32 @test(%struct.A* %obj, i32 %a) comdat {
entry:
  %0 = bitcast %struct.A* %obj to i8**
  %vtable5 = load i8*, i8** %0

  %1 = tail call { i8*, i1 } @llvm.type.checked.load(i8* %vtable5, i32 8, metadata !"_ZTS1A")
  %2 = extractvalue { i8*, i1 } %1, 1
  br i1 %2, label %cont, label %trap

trap:
  tail call void @llvm.trap()
  unreachable

cont:
  %3 = extractvalue { i8*, i1 } %1, 0
  %4 = bitcast i8* %3 to i32 (%struct.A*, i32)*

  %call = tail call i32 %4(%struct.A* nonnull %obj, i32 %a)

  ret i32 %call
}

define linkonce_odr i32 @testb(%struct.A* %obj, i32 %a) comdat {
entry:
  %0 = bitcast %struct.A* %obj to i8**
  %vtable5 = load i8*, i8** %0

  %1 = tail call { i8*, i1 } @llvm.type.checked.load(i8* %vtable5, i32 0, metadata !"_ZTS1A")
  %2 = extractvalue { i8*, i1 } %1, 1
  br i1 %2, label %cont, label %trap

trap:
  tail call void @llvm.trap()
  unreachable

cont:
  %3 = extractvalue { i8*, i1 } %1, 0
  %4 = bitcast i8* %3 to i32 (%struct.A*, i32)*

  %call = tail call i32 %4(%struct.A* nonnull %obj, i32 %a)

  ret i32 %call
}

declare { i8*, i1 } @llvm.type.checked.load(i8*, i32, metadata)
declare void @llvm.trap()

define internal i32 @_ZN1B1fEi(%struct.B* %this, i32 %a) {
entry:
   ret i32 0
}
define internal i32 @_ZN1B1nEi(%struct.B* %this, i32 %a) {
entry:
   ret i32 0
}

!0 = !{i64 16, !"_ZTS1B"}
