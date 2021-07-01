; RUN: opt -passes=openmp-opt -debug-only=openmp-opt -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt -passes=openmp-opt -pass-remarks-analysis=openmp-opt -disable-output < %s 2>&1 | FileCheck %s --check-prefix=REMARKS
; REQUIRES: asserts
; ModuleID = 'single_threaded_exeuction.c'

define weak void @kernel() {
  call void @__kmpc_kernel_init(i32 512, i16 1)
  call void @nvptx()
  call void @amdgcn()
  ret void
}

; REMARKS: remark: single_threaded_execution.c:1:0: Could not internalize function. Some optimizations may not be possible.
; REMARKS-NOT: remark: single_threaded_execution.c:1:0: Could not internalize function. Some optimizations may not be possible.

; CHECK-NOT: [openmp-opt] Basic block @nvptx entry is executed by a single thread.
; CHECK: [openmp-opt] Basic block @nvptx if.then is executed by a single thread.
; CHECK-NOT: [openmp-opt] Basic block @nvptx if.end is executed by a single thread.
; Function Attrs: noinline
define internal void @nvptx() {
entry:
  %call = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %cmp = icmp eq i32 %call, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void @foo()
  call void @bar()
  call void @baz()
  call void @cold()
  br label %if.end

if.end:
  ret void
}

; CHECK-NOT: [openmp-opt] Basic block @amdgcn entry is executed by a single thread.
; CHECK: [openmp-opt] Basic block @amdgcn if.then is executed by a single thread.
; CHECK-NOT: [openmp-opt] Basic block @amdgcn if.end is executed by a single thread.
; Function Attrs: noinline
define internal void @amdgcn() {
entry:
  %call = call i32 @llvm.amdgcn.workitem.id.x()
  %cmp = icmp eq i32 %call, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void @foo()
  call void @bar()
  call void @baz()
  call void @cold()
  br label %if.end

if.end:
  ret void
}

; CHECK: [openmp-opt] Basic block @foo entry is executed by a single thread.
; Function Attrs: noinline
define internal void @foo() {
entry:
  ret void
}

; CHECK: [openmp-opt] Basic block @bar.internalized entry is executed by a single thread.
; Function Attrs: noinline
define void @bar() {
entry:
  ret void
}

; CHECK-NOT: [openmp-opt] Basic block @baz entry is executed by a single thread.
; Function Attrs: noinline
define weak void @baz() !dbg !8 {
entry:
  ret void
}

; CHECK-NOT: [openmp-opt] Basic block @cold entry is executed by a single thread.
; Function Attrs: cold convergent noinline nounwind optnone mustprogress
define weak void @cold() #0 !dbg !9 {
entry:
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

declare i32 @llvm.amdgcn.workitem.id.x()

declare void @__kmpc_kernel_init(i32, i16)

attributes #0 = { cold noinline }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!nvvm.annotations = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "single_threaded_execution.c", directory: "/tmp/single_threaded_execution.c")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"openmp", i32 50}
!6 = !{i32 7, !"openmp-device", i32 50}
!7 = !{void ()* @kernel, !"kernel", i32 1}
!8 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 8, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!9 = distinct !DISubprogram(name: "cold", scope: !1, file: !1, line: 8, type: !10, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !2)
