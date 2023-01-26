; RUN: not mlir-translate -import-llvm -split-input-file %s 2>&1 | FileCheck %s

; CHECK:      import-failure.ll
; CHECK-SAME: error: unhandled instruction: indirectbr ptr %dst, [label %bb1, label %bb2]
define i32 @unhandled_instruction(ptr %dst) {
  indirectbr ptr %dst, [label %bb1, label %bb2]
bb1:
  ret i32 0
bb2:
  ret i32 1
}

; // -----

; CHECK:      import-failure.ll
; CHECK-SAME: error: unhandled value: ptr asm "bswap $0", "=r,r"
define i32 @unhandled_value(i32 %arg1) {
  %1 = call i32 asm "bswap $0", "=r,r"(i32 %arg1)
  ret i32 %1
}

; // -----

; CHECK:      import-failure.ll
; CHECK-SAME: error: blockaddress is not implemented in the LLVM dialect
; CHECK:      import-failure.ll
; CHECK-SAME: error: unhandled instruction: ret ptr blockaddress(@unhandled_constant, %bb1)
define ptr @unhandled_constant() {
bb1:
  ret ptr blockaddress(@unhandled_constant, %bb1)
}

; // -----

; CHECK:      import-failure.ll
; CHECK-SAME: error: blockaddress is not implemented in the LLVM dialect
; CHECK:      import-failure.ll
; CHECK-SAME: error: unhandled global variable: @private = private global ptr blockaddress(@unhandled_global, %bb1)
@private = private global ptr blockaddress(@unhandled_global, %bb1)

define void @unhandled_global() {
bb1:
  ret void
}

; // -----

declare void @llvm.gcroot(ptr %arg1, ptr %arg2)

; CHECK:      import-failure.ll
; CHECK-SAME: error: unhandled intrinsic: call void @llvm.gcroot(ptr %arg1, ptr %arg2)
define void @unhandled_intrinsic(ptr %arg1, ptr %arg2) {
  call void @llvm.gcroot(ptr %arg1, ptr %arg2)
  ret void
}

; // -----

; CHECK:      import-failure.ll
; CHECK-SAME: warning: unhandled metadata: !0 = !{!"unknown metadata"} on br i1 %arg1, label %bb1, label %bb2, !prof !0
define i64 @unhandled_metadata(i1 %arg1, i64 %arg2) {
entry:
  br i1 %arg1, label %bb1, label %bb2, !prof !0
bb1:
  ret i64 %arg2
bb2:
  ret i64 %arg2
}

!0 = !{!"unknown metadata"}

; // -----

; CHECK:      import-failure.ll
; CHECK-SAME: warning: unhandled function metadata: !0 = !{!"unknown metadata"} on define void @unhandled_func_metadata(i1 %arg1, i64 %arg2) !prof !0
define void @unhandled_func_metadata(i1 %arg1, i64 %arg2) !prof !0 {
  ret void
}

!0 = !{!"unknown metadata"}

; // -----

declare void @llvm.dbg.value(metadata, metadata, metadata)

; CHECK:      import-failure.ll
; CHECK-SAME: warning: dropped instruction: call void @llvm.dbg.value(metadata i64 %arg1, metadata !3, metadata !DIExpression(DW_OP_plus_uconst, 42, DW_OP_stack_value)), !dbg !5
; CHECK:      import-failure.ll
; CHECK-SAME: warning: dropped instruction: call void @llvm.dbg.value(metadata !DIArgList(i64 %arg1, i64 undef), metadata !3, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 1, DW_OP_mul, DW_OP_plus, DW_OP_stack_value)), !dbg !5
define void @dropped_instruction(i64 %arg1) {
  call void @llvm.dbg.value(metadata i64 %arg1, metadata !3, metadata !DIExpression(DW_OP_plus_uconst, 42, DW_OP_stack_value)), !dbg !5
  call void @llvm.dbg.value(metadata !DIArgList(i64 %arg1, i64 undef), metadata !3, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 1, DW_OP_mul, DW_OP_plus, DW_OP_stack_value)), !dbg !5
  ret void
}

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2)
!2 = !DIFile(filename: "import-failure.ll", directory: "/")
!3 = !DILocalVariable(scope: !4, name: "arg1", file: !2, line: 1, arg: 1, align: 64);
!4 = distinct !DISubprogram(name: "intrinsic", scope: !2, file: !2, spFlags: DISPFlagDefinition, unit: !1)
!5 = !DILocation(line: 1, column: 2, scope: !4)

; // -----

; global_ctors requires the appending linkage type.
; CHECK:      import-failure.ll
; CHECK-SAME: error: unhandled global variable: @llvm.global_ctors
@llvm.global_ctors = global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @foo, ptr null }]

define void @foo() {
  ret void
}

; // -----

; global_dtors with non-null data fields cannot be represented in MLIR.
; CHECK:      import-failure.ll
; CHECK-SAME: error: unhandled global variable: @llvm.global_dtors
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @foo, ptr @foo }]

define void @foo() {
  ret void
}

; // -----

; global_ctors without a data field should not be imported.
; CHECK:      import-failure.ll
; CHECK-SAME: error: unhandled global variable: @llvm.global_ctors
@llvm.global_ctors = appending global [1 x { i32, ptr }] [{ i32, ptr } { i32 0, ptr @foo }]

define void @foo() {
  ret void
}

; // -----

; global_dtors with a wrong argument order should not be imported.
; CHECK:      import-failure.ll
; CHECK-SAME: error: unhandled global variable: @llvm.global_dtors
@llvm.global_dtors = appending global [1 x { ptr, i32, ptr }] [{ ptr, i32, ptr } { ptr @foo, i32 0, ptr null }]

define void @foo() {
  ret void
}

; // -----

; CHECK:      import-failure.ll
; CHECK-SAME: error: TBAA root node must have non-empty identity: !2 = !{!""}
define dso_local void @tbaa(ptr %0) {
  store i8 1, ptr %0, align 4, !tbaa !2
  ret void
}

!0 = !{!""}
!1 = !{!"omnipotent char", !0, i64 0}
!2 = !{!1, !1, i64 0}

; // -----

; CHECK:      import-failure.ll
; CHECK-SAME: error: unsupported TBAA node format: !0 = !{!1, i64 0, i64 0}
define dso_local void @tbaa(ptr %0) {
  store i8 1, ptr %0, align 4, !tbaa !2
  ret void
}

!0 = !{!"Simple C/C++ TBAA"}
!1 = !{!"omnipotent char", !0, i64 0}
!2 = !{!1, i64 0, i64 0}

; // -----

; CHECK:      import-failure.ll
; CHECK-SAME: error: operand '1' must be MDNode: !1 = !{!"omnipotent char", i64 0, i64 0}
define dso_local void @tbaa(ptr %0) {
  store i8 1, ptr %0, align 4, !tbaa !2
  ret void
}

!0 = !{!"Simple C/C++ TBAA"}
!1 = !{!"omnipotent char", i64 0, i64 0}
!2 = !{!1, !1, i64 0}

; // -----

; CHECK:      import-failure.ll
; CHECK-SAME: error: missing member offset: !1 = !{!"agg_t", !2, i64 0, !2}
define dso_local void @tbaa(ptr %0) {
  store i8 1, ptr %0, align 4, !tbaa !3
  ret void
}

!0 = !{!"Simple C/C++ TBAA"}
!1 = !{!"omnipotent char", !0, i64 0}
!2 = !{!"agg_t", !1, i64 0, !1}
!3 = !{!2, !1, i64 0}

; // -----

; CHECK:      import-failure.ll
; CHECK-SAME: error: operand '4' must be ConstantInt: !1 = !{!"agg_t", !2, i64 0, !2, double 1.000000e+00}
define dso_local void @tbaa(ptr %0) {
  store i8 1, ptr %0, align 4, !tbaa !3
  ret void
}

!0 = !{!"Simple C/C++ TBAA"}
!1 = !{!"omnipotent char", !0, i64 0}
!2 = !{!"agg_t", !1, i64 0, !1, double 1.0}
!3 = !{!2, !1, i64 0}

; // -----

; CHECK:      import-failure.ll
; CHECK-SAME: error: operand '3' must be ConstantInt: !0 = !{!1, !1, i64 0, double 1.000000e+00}
define dso_local void @tbaa(ptr %0) {
  store i8 1, ptr %0, align 4, !tbaa !2
  ret void
}

!0 = !{!"Simple C/C++ TBAA"}
!1 = !{!"omnipotent char", !0, i64 0}
!2 = !{!1, !1, i64 0, double 1.0}

; // -----

; CHECK:      import-failure.ll
; CHECK-SAME: error: unsupported TBAA node format: !0 = !{!1, !1, i64 0, i64 4}
define dso_local void @tbaa(ptr %0) {
  store i32 1, ptr %0, align 4, !tbaa !2
  ret void
}

!2 = !{!3, !3, i64 0, i64 4}
!3 = !{!4, i64 4, !"int"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}
