; Tests that asan skips pre-split coroutine and NoopCoro.Frame
; RUN: opt < %s -S -passes=coro-early,asan | FileCheck %s

; CHECK: %NoopCoro.Frame = type { ptr, ptr }
; CHECK: @NoopCoro.Frame.Const = private constant %NoopCoro.Frame { ptr @__NoopCoro_ResumeDestroy, ptr @__NoopCoro_ResumeDestroy }
; CHECK-NOT: @0 = private alias { %NoopCoro.Frame,

%struct.Promise = type { %"struct.std::__n4861::coroutine_handle" }
%"struct.std::__n4861::coroutine_handle" = type { ptr }

; CHECK-LABEL: @foo(
define ptr @foo() #0 {
; CHECK-NEXT: entry
; CHECK-NOT: %asan_local_stack_base
entry:
  %__promise = alloca %struct.Promise, align 8
  %0 = call token @llvm.coro.id(i32 16, ptr nonnull %__promise, ptr null, ptr null)
  %1 = call ptr @llvm.coro.noop()
  ret ptr %1
}

declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr)
declare ptr @llvm.coro.noop()

attributes #0 = { sanitize_address presplitcoroutine }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "hand-written", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
