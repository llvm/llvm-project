; RUN: not mlir-translate -import-llvm -split-input-file %s 2>&1 | FileCheck %s

; CHECK: error: unhandled instruction indirectbr i8* %dst, [label %bb1, label %bb2]
define i32 @unhandled_instruction(i8* %dst) {
  indirectbr i8* %dst, [label %bb1, label %bb2]
bb1:
  ret i32 0
bb2:
  ret i32 1
}

; // -----

; CHECK: unhandled value ptr asm "bswap $0", "=r,r"
define i32 @unhandled_value(i32 %arg0) {
  %1 = call i32 asm "bswap $0", "=r,r"(i32 %arg0)
  ret i32 %1
}

; // -----

; CHECK: error: unhandled constant i8* blockaddress(@unhandled_constant, %bb1)
define i8* @unhandled_constant() {
bb1:
  ret i8* blockaddress(@unhandled_constant, %bb1)
}

; // -----

; CHECK: error: unhandled constant i8* blockaddress(@unhandled_global, %bb1)
; CHECK: error: unhandled global variable @private = private global i8* blockaddress(@unhandled_global, %bb1)
@private = private global i8* blockaddress(@unhandled_global, %bb1)

define void @unhandled_global() {
bb1:
  ret void
}

; // -----

declare void @llvm.gcroot(ptr %arg0, ptr %arg1)

; CHECK: error: unhandled intrinsic call void @llvm.gcroot(ptr %arg0, ptr %arg1)
define void @unhandled_intrinsic(ptr %arg0, ptr %arg1) {
  call void @llvm.gcroot(ptr %arg0, ptr %arg1)
  ret void
}

; // -----

; CHECK: warning: unhandled metadata (2)   br i1 %arg1, label %bb1, label %bb2, !prof !0
define i64 @cond_br(i1 %arg1, i64 %arg2) {
entry:
  br i1 %arg1, label %bb1, label %bb2, !prof !0
bb1:
  ret i64 %arg2
bb2:
  ret i64 %arg2
}

!0 = !{!"unknown metadata"}

; // -----

declare void @llvm.dbg.value(metadata, metadata, metadata)

; CHECK: warning: dropped instruction   call void @llvm.dbg.value(metadata i64 %arg1, metadata !3, metadata !DIExpression(DW_OP_plus_uconst, 42, DW_OP_stack_value)), !dbg !5
define void @dropped_instruction(i64 %arg1) {
  call void @llvm.dbg.value(metadata i64 %arg1, metadata !3, metadata !DIExpression(DW_OP_plus_uconst, 42, DW_OP_stack_value)), !dbg !5
  ret void
}

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2)
!2 = !DIFile(filename: "import-failure.ll", directory: "/")
!3 = !DILocalVariable(scope: !4, name: "arg", file: !2, line: 1, arg: 1, align: 32);
!4 = distinct !DISubprogram(name: "intrinsic", scope: !2, file: !2, spFlags: DISPFlagDefinition, unit: !1)
!5 = !DILocation(line: 1, column: 2, scope: !4)

; // -----

; global_ctors requires the appending linkage type.
; CHECK: error: unhandled global variable @llvm.global_ctors
@llvm.global_ctors = global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @foo, ptr null }]

define void @foo() {
  ret void
}

; // -----

; global_dtors with non-null data fields cannot be represented in MLIR.
; CHECK: error: unhandled global variable @llvm.global_dtors
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @foo, ptr @foo }]

define void @foo() {
  ret void
}

; // -----

; global_ctors without a data field should not be imported.
; CHECK: error: unhandled global variable @llvm.global_ctors
@llvm.global_ctors = appending global [1 x { i32, ptr }] [{ i32, ptr } { i32 0, ptr @foo }]

define void @foo() {
  ret void
}

; // -----

; global_dtors with a wrong argument order should not be imported.
; CHECK: error: unhandled global variable @llvm.global_dtors
@llvm.global_dtors = appending global [1 x { ptr, i32, ptr }] [{ ptr, i32, ptr } { ptr @foo, i32 0, ptr null }]

define void @foo() {
  ret void
}
