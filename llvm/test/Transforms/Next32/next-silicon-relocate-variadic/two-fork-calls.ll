; RUN: opt -S -passes=next-silicon-relocate-variadic %s | FileCheck %s

; CHECK-LABEL: define i32 @main
; CHECK-NEXT:   [[TMP1:%.*]] = alloca [1000 x i32], align 16
; CHECK-NEXT:   call void @__next_boundary_thunk1(ptr [[TMP1]], i32 1, ptr [[TMP1]])
; CHECK-NEXT:   call void @__next_boundary_thunk2(ptr [[TMP1]], i32 1, ptr [[TMP1]])
; CHECK-NEXT:   ret i32 0
define i32 @main() !dbg !5 {
  %1 = alloca [1000 x i32], align 16
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr %1, i32 1, ptr @.omp_outlined., ptr %1), !dbg !6
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr %1, i32 1, ptr @.omp_outlined.2, ptr %1), !dbg !7
  ret i32 0
}

; CHECK-LABEL: define void @.omp_outlined.
define void @.omp_outlined.(ptr %0, ptr %1, ptr nocapture %2) !dbg !8 {
  %4 = alloca i32, align 4
  store i32 0, ptr %4, align 4
  %5 = load i32, ptr %0, align 4
  call void @__kmpc_for_static_init_4(ptr %0, i32 %5, i32 34, ptr %4, ptr %4, ptr %4, ptr %4, i32 1, i32 1)
  call void @__kmpc_for_static_fini(ptr %0, i32 %5)
  ret void
}

; CHECK-LABEL: define internal void @__next_boundary_thunk1
; CHECK-SAME: ptr [[TMP0:%.*]], i32 [[TMP1:%.*]], ptr nocapture [[TMP2:%.*]])
; CHECK-SAME: #[[BOUNDARY_ATTR:[0-9]+]]
; CHECK-SAME: !dbg !{{[0-9]+}}
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @__next_variadic_callsite_thunk1(ptr [[TMP0]], i32 [[TMP1]], ptr @.omp_outlined., ptr [[TMP2]], ptr @__kmpc_fork_call), !dbg !{{[0-9]+}}
; CHECK-NEXT:    ret void

; CHECK-LABEL: define void @.omp_outlined.2
define void @.omp_outlined.2(ptr %0, ptr %1, ptr noalias %2) !dbg !9 {
  %4 = alloca i32, align 4
  store i32 0, ptr %4, align 4
  %5 = load i32, ptr %0, align 4
  call void @__kmpc_for_static_init_4(ptr %0, i32 %5, i32 34, ptr %4, ptr %4, ptr %4, ptr %4, i32 1, i32 1)
  call void @__kmpc_for_static_fini(ptr %0, i32 %5)
  ret void
}

; CHECK-LABEL: define internal void @__next_boundary_thunk2
; CHECK-SAME: ptr [[TMP0:%.*]], i32 [[TMP1:%.*]], ptr noalias [[TMP2:%.*]])
; CHECK-SAME: #[[BOUNDARY_ATTR]]
; CHECK-SAME: !dbg !{{[0-9]+}}
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @__next_variadic_callsite_thunk1(ptr [[TMP0]], i32 [[TMP1]], ptr @.omp_outlined.2, ptr [[TMP2]], ptr @__kmpc_fork_call), !dbg !{{[0-9]+}}
; CHECK-NEXT:    ret void

; CHECK: attributes #[[BOUNDARY_ATTR]] = { noinline "ns-boundary-function" }

declare void @__kmpc_fork_call(ptr, i32, ptr, ...) local_unnamed_addr

declare void @__kmpc_for_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32) local_unnamed_addr

declare void @__kmpc_for_static_fini(ptr, i32) local_unnamed_addr

!llvm.dbg.cu = !{!0}
!llvm.ident = !{!2}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "NextSilicon clang version 16.0.4 (git@github.com:nextsilicon/next-llvm-project.git)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "omp_lit.c", directory: "/space2/users/ilias/sw/ex")
!2 = !{!"NextSilicon clang version 16.0.4 (git@github.com:nextsilicon/next-llvm-project.git 8f2830cab3d60993a39c6c67df49792cf2572c54)"}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: ".main", scope: !1, file: !1, line: 4, unit: !0)
!6 = !DILocation(line: 4, column: 8, scope: !5)
!7 = !DILocation(line: 5, column: 8, scope: !5)
!8 = distinct !DISubprogram(name: ".omp_outlined.", scope: !1, file: !1, line: 4, unit: !0)
!9 = distinct !DISubprogram(name: ".omp_outlined.2", scope: !1, file: !1, line: 4, unit: !0)
