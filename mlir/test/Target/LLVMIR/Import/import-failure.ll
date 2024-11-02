; RUN: not mlir-translate -import-llvm -split-input-file %s 2>&1 | FileCheck %s

; CHECK: unhandled instruction indirectbr i8* %dst, [label %bb1, label %bb2]
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

; CHECK: unhandled constant i8* blockaddress(@unhandled_constant, %bb1)
define i8* @unhandled_constant() {
bb1:
  ret i8* blockaddress(@unhandled_constant, %bb1)
}

; // -----

declare void @llvm.gcroot(ptr %arg0, ptr %arg1)

; CHECK: unhandled intrinsic call void @llvm.gcroot(ptr %arg0, ptr %arg1)
define void @unhandled_intrinsic(ptr %arg0, ptr %arg1) {
  call void @llvm.gcroot(ptr %arg0, ptr %arg1)
  ret void
}
