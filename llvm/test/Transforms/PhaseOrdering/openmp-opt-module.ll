; RUN: opt -passes='default<O2>' -pass-remarks-missed=openmp-opt < %s 2>&1 | FileCheck %s --check-prefix=MODULE
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"

%struct.ident_t = type { i32, i32, i32, i32, i8* }

@.str = private unnamed_addr constant [13 x i8] c"Alloc Shared\00", align 1

@S = external local_unnamed_addr global i8*

; MODULE: remark: openmp_opt_module.c:5:7: Found thread data sharing on the GPU. Expect degraded performance due to data globalization.

define void @foo() {
entry:
  %i = call i32 @__kmpc_target_init(%struct.ident_t* null, i1 false, i1 true, i1 true)
  %x = call i8* @__kmpc_alloc_shared(i64 4), !dbg !10
  %x_on_stack = bitcast i8* %x to i32*
  %0 = bitcast i32* %x_on_stack to i8*
  call void @use(i8* %0)
  call void @__kmpc_free_shared(i8* %x)
  call void @__kmpc_target_deinit(%struct.ident_t* null, i1 false, i1 true)
  ret void
}

declare void @use(i8* %0)

define weak i8* @__kmpc_alloc_shared(i64 %DataSize) {
entry:
  %call = call i8* @_Z10SafeMallocmPKc(i64 %DataSize, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i64 0, i64 0)) #11
  ret i8* %call
}

; Function Attrs: convergent nounwind mustprogress
declare i8* @_Z10SafeMallocmPKc(i64 %size, i8* nocapture readnone %msg)

declare void @__kmpc_free_shared(i8*)
declare i32 @__kmpc_target_init(%struct.ident_t*, i1, i1 %use_generic_state_machine, i1)
declare void @__kmpc_target_deinit(%struct.ident_t*, i1, i1)

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
!7 = !{void ()* @foo, !"kernel", i32 1}
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !2)
!10 = !DILocation(line: 5, column: 7, scope: !8)
