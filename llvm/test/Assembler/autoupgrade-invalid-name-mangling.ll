; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: Intrinsic called with incompatible signature
; CHECK-NEXT: %reg = call i32 @llvm.read_register.i64(
; CHECK: Invalid user of intrinsic instruction!
; CHECK-NEXT: %reg = call i32 @llvm.read_register.i64(
define i32 @read_register_missing_mangling() {
  %reg = call i32 @llvm.read_register(metadata !0)
  ret i32 %reg
}

declare i64 @llvm.read_register(metadata)

!0 = !{!"foo"}
