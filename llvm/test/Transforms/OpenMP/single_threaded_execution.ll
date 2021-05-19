; RUN: opt -passes=openmp-opt-cgscc -debug-only=openmp-opt -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts
; ModuleID = 'single_threaded_exeuction.c'

define void @kernel() {
  call void @__kmpc_kernel_init(i32 512, i16 1)
  call void @nvptx()
  call void @amdgcn()
  ret void
}

; CHECK-NOT: [openmp-opt] Basic block @nvptx entry is executed by a single thread.
; CHECK: [openmp-opt] Basic block @nvptx if.then is executed by a single thread.
; CHECK-NOT: [openmp-opt] Basic block @nvptx if.end is executed by a single thread.
; Function Attrs: noinline nounwind uwtable
define dso_local void @nvptx() {
entry:
  %call = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %cmp = icmp eq i32 %call, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void @bar()
  br label %if.end

if.end:
  ret void
}

; CHECK-NOT: [openmp-opt] Basic block @amdgcn entry is executed by a single thread.
; CHECK: [openmp-opt] Basic block @amdgcn if.then is executed by a single thread.
; CHECK-NOT: [openmp-opt] Basic block @amdgcn if.end is executed by a single thread.
; Function Attrs: noinline nounwind uwtable
define dso_local void @amdgcn() {
entry:
  %call = call i32 @llvm.amdgcn.workitem.id.x()
  %cmp = icmp eq i32 %call, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void @bar()
  br label %if.end

if.end:
  ret void
}

; CHECK: [openmp-opt] Basic block @bar entry is executed by a single thread.
; Function Attrs: noinline nounwind uwtable
define internal void @bar() {
entry:
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

declare i32 @llvm.amdgcn.workitem.id.x()

declare void @__kmpc_kernel_init(i32, i16)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!nvvm.annotations = !{!5}


!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "single_threaded_execution.c", directory: "/tmp/single_threaded_execution.c")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{void ()* @kernel, !"kernel", i32 1}
