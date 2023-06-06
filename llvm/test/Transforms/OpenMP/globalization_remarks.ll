; RUN: opt -passes=openmp-opt -pass-remarks-missed=openmp-opt -disable-output < %s 2>&1 | FileCheck %s
; ModuleID = 'declare_target_codegen_globalization.cpp'
source_filename = "declare_target_codegen_globalization.cpp"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64"

; CHECK: remark: globalization_remarks.c:5:7: Could not move globalized variable to the stack. Variable is potentially captured in call. Mark parameter as `__attribute__((noescape))` to override.
; CHECK: remark: globalization_remarks.c:5:7: Found thread data sharing on the GPU. Expect degraded performance due to data globalization.

%struct.ident_t = type { i32, i32, i32, i32, ptr }

@S = external local_unnamed_addr global ptr

define void @foo() "kernel" {
entry:
  %c = call i32 @__kmpc_target_init(ptr null, i1 false, i1 true)
  %0 = call ptr @__kmpc_alloc_shared(i64 4), !dbg !10
  call void @share(ptr %0), !dbg !10
  call void @__kmpc_free_shared(ptr %0)
  call void @__kmpc_target_deinit(ptr null, i1 false)
  ret void
}

define internal void @share(ptr %x) {
entry:
  store ptr %x, ptr @S
  ret void
}

declare ptr @__kmpc_alloc_shared(i64)

declare void @__kmpc_free_shared(ptr nocapture)

declare i32 @__kmpc_target_init(ptr, i1, i1);

declare void @__kmpc_target_deinit(ptr, i1)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!nvvm.annotations = !{!7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "globalization_remarks.c", directory: "/tmp/globalization_remarks.c")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"openmp", i32 50}
!6 = !{i32 7, !"openmp-device", i32 50}
!7 = !{ptr @foo, !"kernel", i32 1}
!8 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !2)
!10 = !DILocation(line: 5, column: 7, scope: !8)
