; RUN: opt -passes=openmp-opt-cgscc -debug-only=openmp-opt -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts
; ModuleID = 'single_threaded_exeuction.c'

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @0, i32 0, i32 0) }, align 8

; CHECK: [openmp-opt] Basic block @bar entry is executed by a single thread.
; Function Attrs: noinline nounwind uwtable
define internal void @bar() {
entry:
  ret void
}

; CHECK-NOT: [openmp-opt] Basic block @foo entry is executed by a single thread.
; CHECK: [openmp-opt] Basic block @foo if.then is executed by a single thread.
; CHECK-NOT: [openmp-opt] Basic block @foo if.end is executed by a single thread.
; Function Attrs: noinline nounwind uwtable
define dso_local void @foo() {
entry:
  %call = call i32 @omp_get_thread_num()
  %cmp = icmp eq i32 %call, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void @bar()
  br label %if.end

if.end:
  ret void
}

declare dso_local i32 @omp_get_thread_num()

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 13.0.0"}
!2 = !{!3}
!3 = !{i64 2, i64 -1, i64 -1, i1 true}
