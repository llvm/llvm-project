; ModuleID = 'foo.cpp'
source_filename = "foo.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { i32 (...)** }

; Function Attrs: uwtable
define hidden i32 @_Z3fooP1A(%struct.A* %pA) local_unnamed_addr {
entry:
  %0 = bitcast %struct.A* %pA to i32 (%struct.A*)***
  %vtable = load i32 (%struct.A*)**, i32 (%struct.A*)*** %0, align 8, !tbaa !2
  %1 = bitcast i32 (%struct.A*)** %vtable to i8*
  %2 = tail call i1 @llvm.type.test(i8* %1, metadata !"_ZTS1A")
  tail call void @llvm.assume(i1 %2)
  %3 = load i32 (%struct.A*)*, i32 (%struct.A*)** %vtable, align 8
  %call = tail call i32 %3(%struct.A* %pA)
  %add = add nsw i32 %call, 10
  ret i32 %add
}

; Function Attrs: nounwind readnone willreturn
declare i1 @llvm.type.test(i8*, metadata)

; Function Attrs: nounwind willreturn
declare void @llvm.assume(i1)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 (trunk 373596)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"vtable pointer", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
