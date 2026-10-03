; Argument tracing: one __sanitizer_cov_trace_args call per source-level
; parameter. A scalar is reported as its value, a pointer to a struct as the
; address of that struct together with its field offset table, and hidden ABI
; arguments are left out.
;
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-args -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i32, i64 }

; struct S { int a; long b; } is described by one {byte offset, byte size} pair
; per field. The table is shared by every value of that type, so both functions
; below referring to struct S must produce exactly one.
; CHECK: @__sancov_offsets_ = private unnamed_addr constant [4 x i64] [i64 0, i64 4, i64 8, i64 8]
; CHECK-NOT: = private unnamed_addr constant [{{.*}} x i64]

; void two_params(struct S *s, int x)
define void @two_params(ptr %s, i32 %x) !dbg !11 {
entry:
    #dbg_value(ptr %s, !15, !DIExpression(), !17)
    #dbg_value(i32 %x, !16, !DIExpression(), !17)
  ret void
}
; The pointer is reported as the address of the 16-byte struct it points at, so
; that a consumer can read the two fields out of it. Nothing is spilled.
; CHECK-LABEL: define void @two_params(
; CHECK-NOT: alloca
; CHECK: %[[ADDR:[0-9]+]] = ptrtoint ptr %s to i64
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @two_params to i64), i32 0, i32 16, i64 %[[ADDR]], ptr @__sancov_offsets_, i32 2)
; The scalar is reported as its own value, widened to 64 bits.
; CHECK: %[[X:[0-9]+]] = zext i32 %x to i64
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @two_params to i64), i32 1, i32 4, i64 %[[X]], ptr null, i32 0)

; struct S sret_and_scalar(int x), returning the struct through a hidden
; struct-return pointer: that pointer is not a source-level parameter, so `x`
; keeps source index 0 rather than becoming the IR argument index 1.
define void @sret_and_scalar(ptr sret(%struct.S) %0, i32 %1) !dbg !18 {
entry:
    #dbg_value(i32 %1, !22, !DIExpression(), !23)
  ret void
}
; CHECK-LABEL: define void @sret_and_scalar(
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @sret_and_scalar to i64), i32 0, i32 4, i64 %{{[0-9]+}}, ptr null, i32 0)
; CHECK-NOT: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @sret_and_scalar to i64), i32 1

; void dead_param(struct S *s, int x) with `x` optimized away: it has no debug
; record left, but the subprogram still declares it, so it is reported with
; size 0 to keep the parameter list complete.
define void @dead_param(ptr %s) !dbg !24 {
entry:
    #dbg_value(ptr %s, !28, !DIExpression(), !29)
  ret void
}
; CHECK-LABEL: define void @dead_param(
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @dead_param to i64), i32 0, i32 16, i64 %{{[0-9]+}}, ptr @__sancov_offsets_, i32 2)
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @dead_param to i64), i32 1, i32 0, i64 0, ptr null, i32 0)

; A pointer whose pointee has no field layout - void *, int * - is reported as
; the pointer value itself rather than as an address to read through.
define void @opaque_pointer(ptr %p) !dbg !30 {
entry:
    #dbg_value(ptr %p, !32, !DIExpression(), !33)
  ret void
}
; CHECK-LABEL: define void @opaque_pointer(
; CHECK: %[[P:[0-9]+]] = ptrtoint ptr %p to i64
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @opaque_pointer to i64), i32 0, i32 8, i64 %[[P]], ptr null, i32 0)

; A floating-point parameter is reported as its bit pattern.
define void @floating(double %d) !dbg !34 {
entry:
    #dbg_value(double %d, !36, !DIExpression(), !37)
  ret void
}
; CHECK-LABEL: define void @floating(
; CHECK: %[[BITS:[0-9]+]] = bitcast double %d to i64
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @floating to i64), i32 0, i32 8, i64 %[[BITS]], ptr null, i32 0)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, isOptimized: true, emissionKind: FullDebug)
!1 = !{i32 2, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "args.c", directory: "/")
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, size: 128, elements: !7)
!7 = !{!8, !9}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !6, file: !3, baseType: !4, size: 32)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !6, file: !3, baseType: !5, size: 64, offset: 64)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)

!11 = distinct !DISubprogram(name: "two_params", scope: !3, file: !3, line: 1, type: !12, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !10, !4}
!14 = !{!15, !16}
!15 = !DILocalVariable(name: "s", arg: 1, scope: !11, file: !3, line: 1, type: !10)
!16 = !DILocalVariable(name: "x", arg: 2, scope: !11, file: !3, line: 1, type: !4)
!17 = !DILocation(line: 1, column: 1, scope: !11)

!18 = distinct !DISubprogram(name: "sret_and_scalar", scope: !3, file: !3, line: 5, type: !19, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !21)
!19 = !DISubroutineType(types: !20)
!20 = !{!6, !4}
!21 = !{!22}
!22 = !DILocalVariable(name: "x", arg: 1, scope: !18, file: !3, line: 5, type: !4)
!23 = !DILocation(line: 5, column: 1, scope: !18)

!24 = distinct !DISubprogram(name: "dead_param", scope: !3, file: !3, line: 9, type: !25, scopeLine: 9, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !26)
!25 = !DISubroutineType(types: !13)
!26 = !{!28}
!28 = !DILocalVariable(name: "s", arg: 1, scope: !24, file: !3, line: 9, type: !10)
!29 = !DILocation(line: 9, column: 1, scope: !24)

!30 = distinct !DISubprogram(name: "opaque_pointer", scope: !3, file: !3, line: 13, type: !38, scopeLine: 13, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !31)
!31 = !{!32}
!32 = !DILocalVariable(name: "p", arg: 1, scope: !30, file: !3, line: 13, type: !39)
!33 = !DILocation(line: 13, column: 1, scope: !30)

!34 = distinct !DISubprogram(name: "floating", scope: !3, file: !3, line: 17, type: !40, scopeLine: 17, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !35)
!35 = !{!36}
!36 = !DILocalVariable(name: "d", arg: 1, scope: !34, file: !3, line: 17, type: !42)
!37 = !DILocation(line: 17, column: 1, scope: !34)

!38 = !DISubroutineType(types: !{null, !39})
!39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)
!40 = !DISubroutineType(types: !{null, !42})
!42 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
