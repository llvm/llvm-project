; RUN: opt -passes='default<O2>' -pass-remarks-missed=openmp-opt < %s 2>&1 | FileCheck %s --check-prefix=MODULE
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"

%struct.ConfigurationEnvironmentTy = type { i8, i8, i8 }
%struct.KernelEnvironmentTy = type { %struct.ConfigurationEnvironmentTy, ptr, ptr }

@.str = private unnamed_addr constant [13 x i8] c"Alloc Shared\00", align 1
@S = external local_unnamed_addr global ptr
@foo_kernel_environment = local_unnamed_addr constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 1, i8 0, i8 1 }, ptr null, ptr null }

; MODULE: remark: openmp_opt_module.c:5:7: Found thread data sharing on the GPU. Expect degraded performance due to data globalization.

define void @foo() "kernel" {
entry:
  %i = call i32 @__kmpc_target_init(ptr @foo_kernel_environment)
  %x = call ptr @__kmpc_alloc_shared(i64 4), !dbg !10
  call void @use(ptr %x)
  call void @__kmpc_free_shared(ptr %x)
  call void @__kmpc_target_deinit()
  ret void
}

declare void @use(ptr %x)

define weak ptr @__kmpc_alloc_shared(i64 %DataSize) {
entry:
  %call = call ptr @_Z10SafeMallocmPKc(i64 %DataSize, ptr @.str) #11
  ret ptr %call
}

; Function Attrs: convergent nounwind mustprogress
declare ptr @_Z10SafeMallocmPKc(i64 %size, ptr nocapture readnone %msg)

declare void @__kmpc_free_shared(ptr)
declare i32 @__kmpc_target_init(ptr)
declare void @__kmpc_target_deinit()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!nvvm.annotations = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "openmp_opt_module.c", directory: "/tmp/openmp_opt_module.c")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"openmp", i32 50}
!6 = !{i32 7, !"openmp-device", i32 50}
!7 = !{ptr @foo, !"kernel", i32 1}
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !2)
!10 = !DILocation(line: 5, column: 7, scope: !8)
