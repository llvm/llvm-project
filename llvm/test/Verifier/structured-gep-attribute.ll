; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

target datalayout = "e-p:32:32:32-p1:16:16:16-p2:8:8:8-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n8:16:32"

%S = type { i32, i32 }

@global = global %S zeroinitializer

; CHECK: Intrinsic first parameter is missing an ElementType attribute

define void @foo() {
entry:
  %ptr = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr @global, i32 0)
  ret void
}
