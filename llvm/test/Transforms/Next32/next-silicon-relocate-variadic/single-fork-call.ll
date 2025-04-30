; RUN: opt -S -passes=next-silicon-relocate-variadic %s | FileCheck %s

declare void @__kmpc_fork_call(ptr, i32, ptr, ...)

declare void @__kmpc_for_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32)

declare void @__kmpc_for_static_fini(ptr, i32)

; CHECK-LABEL: define i32 @main
; CHECK-NEXT:    [[TMP1:%.*]] = alloca [1000 x i32], align 16
; CHECK-NEXT:    call void @__next_boundary_thunk1(ptr [[TMP1]], i32 2, ptr [[TMP1]], i32 42)
; CHECK-NEXT:    ret i32 0
define i32 @main() "ns-location"="grid" !dbg !5 {
  %1 = alloca [1000 x i32], align 16
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr nonnull %1, i32 2, ptr nonnull @.omp_outlined., ptr nonnull %1, i32 42), !dbg !6
  ret i32 0
}

; CHECK-LABEL: define void @.omp_outlined.
define void @.omp_outlined.(ptr %0, ptr %1, ptr noalias nocapture %2, i32 noundef %3) !dbg !7 {
  %5 = alloca i32, align 4
  %6 = load i32, ptr %0, align 4
  call void @__kmpc_for_static_init_4(ptr nonnull %0, i32 %6, i32 34, ptr nonnull %5, ptr nonnull %5, ptr nonnull %5, ptr nonnull %5, i32 1, i32 1)
  call void @__kmpc_for_static_fini(ptr nonnull %0, i32 %6)
  ret void
}

; CHECK-LABEL: define internal void @__next_boundary_thunk1
; CHECK-SAME: ptr [[TMP0:%.*]], i32 [[TMP1:%.*]], ptr noalias nocapture [[TMP2:%.*]], i32 noundef [[TMP3:%.*]])
; CHECK-SAME: #[[BOUNDARY_ATTR:[0-9]+]]
; CHECK-SAME: !dbg !{{[0-9]+}}
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @__next_variadic_callsite_thunk1(ptr [[TMP0]], i32 [[TMP1]], ptr @.omp_outlined., ptr [[TMP2]], i32 [[TMP3]], ptr @__kmpc_fork_call), !dbg !{{[0-9]+}}
; CHECK-NEXT:    ret void

; CHECK: attributes #[[BOUNDARY_ATTR]] = { noinline "ns-boundary-function" }

!llvm.dbg.cu = !{!0}
!llvm.ident = !{!2}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1)
!1 = !DIFile(filename: "omp_lit.c", directory: "/")
!2 = !{!"NextSilicon clang version 16.0.4 (git@github.com:nextsilicon/next-llvm-project.git 8f2830cab3d60993a39c6c67df49792cf2572c54)"}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: ".main", scope: !1, file: !1, line: 4, unit: !0)
!6 = !DILocation(line: 5, column: 8, scope: !5)
!7 = distinct !DISubprogram(name: ".omp_outlined.", scope: !1, file: !1, line: 4, unit: !0)
