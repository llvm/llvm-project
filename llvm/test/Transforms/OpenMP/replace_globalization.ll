; RUN: opt -S -passes='openmp-opt' < %s | FileCheck %s
; RUN: opt -passes=openmp-opt -pass-remarks=openmp-opt -disable-output < %s 2>&1 | FileCheck %s -check-prefix=CHECK-REMARKS
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64"

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@S = external local_unnamed_addr global i8*
@0 = private unnamed_addr constant [113 x i8] c";llvm/test/Transforms/OpenMP/custom_state_machines_remarks.c;__omp_offloading_2a_d80d3d_test_fallback_l11;11;1;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([113 x i8], [113 x i8]* @0, i32 0, i32 0) }, align 8

; CHECK-REMARKS: remark: replace_globalization.c:5:7: Replaced globalized variable with 16 bytes of shared memory
; CHECK-REMARKS: remark: replace_globalization.c:5:14: Replaced globalized variable with 4 bytes of shared memory
; CHECK-REMARKS-NOT: 6 bytes

; CHECK: [[SHARED_X:@.+]] = internal addrspace(3) global [16 x i8] undef
; CHECK: [[SHARED_Y:@.+]] = internal addrspace(3) global [4 x i8] undef

; CHECK: %{{.*}} = call i8* @__kmpc_alloc_shared({{.*}})
; CHECK: call void @__kmpc_free_shared({{.*}})
define dso_local void @foo() {
entry:
  %c = call i32 @__kmpc_target_init(%struct.ident_t* @1, i1 false, i1 true, i1 true)
  %x = call i8* @__kmpc_alloc_shared(i64 4)
  %x_on_stack = bitcast i8* %x to i32*
  %0 = bitcast i32* %x_on_stack to i8*
  call void @use(i8* %0)
  call void @__kmpc_free_shared(i8* %x, i64 4)
  call void @__kmpc_target_deinit(%struct.ident_t* @1, i1 false, i1 true)
  ret void
}

define void @bar() {
  %c = call i32 @__kmpc_target_init(%struct.ident_t* @1, i1 false, i1 true, i1 true)
  call void @baz()
  call void @qux()
  call void @negative_qux_spmd()
  call void @__kmpc_target_deinit(%struct.ident_t* @1, i1 false, i1 true)
  ret void
}

; CHECK: call void @use.internalized(i8* nofree writeonly addrspacecast (i8 addrspace(3)* getelementptr inbounds ([16 x i8], [16 x i8] addrspace(3)* [[SHARED_X]], i32 0, i32 0) to i8*))
define internal void @baz() {
entry:
  %call = call i32 @__kmpc_target_init(%struct.ident_t* nonnull @1, i1 false, i1 false, i1 true)
  %cmp = icmp eq i32 %call, -1
  br i1 %cmp, label %master, label %exit
master:
  %x = call i8* @__kmpc_alloc_shared(i64 16), !dbg !11
  %x_on_stack = bitcast i8* %x to [4 x i32]*
  %0 = bitcast [4 x i32]* %x_on_stack to i8*
  call void @use(i8* %0)
  call void @__kmpc_free_shared(i8* %x, i64 16)
  br label %exit
exit:
  ret void
}

; CHECK: call void @use.internalized(i8* nofree writeonly addrspacecast (i8 addrspace(3)* getelementptr inbounds ([4 x i8], [4 x i8] addrspace(3)* [[SHARED_Y]], i32 0, i32 0) to i8*))
define internal void @qux() {
entry:
  %call = call i32 @__kmpc_target_init(%struct.ident_t* nonnull @1, i1 false, i1 true, i1 true)
  %0 = icmp eq i32 %call, -1
  br i1 %0, label %master, label %exit
master:
  %y = call i8* @__kmpc_alloc_shared(i64 4), !dbg !12
  %y_on_stack = bitcast i8* %y to [4 x i32]*
  %1 = bitcast [4 x i32]* %y_on_stack to i8*
  call void @use(i8* %1)
  call void @__kmpc_free_shared(i8* %y, i64 4)
  br label %exit
exit:
  ret void
}

define internal void @negative_qux_spmd() {
entry:
  %call = call i32 @__kmpc_target_init(%struct.ident_t* nonnull @1, i1 true, i1 true, i1 true)
  %0 = icmp eq i32 %call, -1
  br i1 %0, label %master, label %exit
master:
  %y = call i8* @__kmpc_alloc_shared(i64 24), !dbg !12
  %y_on_stack = bitcast i8* %y to [6 x i32]*
  %1 = bitcast [6 x i32]* %y_on_stack to i8*
  call void @use(i8* %1)
  call void @__kmpc_free_shared(i8* %y, i64 24)
  br label %exit
exit:
  ret void
}


define void @use(i8* %x) {
entry:
  store i8* %x, i8** @S
  ret void
}

declare i8* @__kmpc_alloc_shared(i64)

declare void @__kmpc_free_shared(i8*, i64)

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

declare i32 @llvm.nvvm.read.ptx.sreg.warpsize()

declare i32 @__kmpc_target_init(%struct.ident_t*, i1, i1, i1)

declare void @__kmpc_target_deinit(%struct.ident_t*, i1, i1)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!nvvm.annotations = !{!7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "replace_globalization.c", directory: "/tmp/replace_globalization.c")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"openmp", i32 50}
!6 = !{i32 7, !"openmp-device", i32 50}
!7 = !{void ()* @foo, !"kernel", i32 1}
!8 = !{void ()* @bar, !"kernel", i32 1}
!9 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !2)
!11 = !DILocation(line: 5, column: 7, scope: !9)
!12 = !DILocation(line: 5, column: 14, scope: !9)
