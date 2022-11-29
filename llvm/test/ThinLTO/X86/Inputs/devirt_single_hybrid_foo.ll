; ModuleID = 'foo.cpp'
source_filename = "foo.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { ptr }

; Function Attrs: uwtable
define hidden i32 @_Z3fooP1A(ptr %pA) local_unnamed_addr {
entry:
  %vtable = load ptr, ptr %pA, align 8, !tbaa !2
  %0 = tail call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS1A")
  tail call void @llvm.assume(i1 %0)
  %1 = load ptr, ptr %vtable, align 8
  %call = tail call i32 %1(ptr %pA)
  %add = add nsw i32 %call, 10
  ret i32 %add
}

; Function Attrs: nounwind readnone willreturn
declare i1 @llvm.type.test(ptr, metadata)

; Function Attrs: nounwind willreturn
declare void @llvm.assume(i1)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 (trunk 373596)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"vtable pointer", !4, i64 0}
!4 = !{!"Simple C++ TBAA"}
