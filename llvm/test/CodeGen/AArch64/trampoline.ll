; RUN: llc -mtriple=aarch64-- < %s | FileCheck %s

declare void @llvm.init.trampoline(ptr, ptr, ptr);
declare ptr @llvm.adjust.trampoline(ptr);

define i64 @f(ptr nest %c, i64 %x, i64 %y) {
  %sum = add i64 %x, %y
  ret i64 %sum
}

define i64 @main() {
  %val = alloca i64
  %nval = bitcast ptr %val to ptr
  %tramp = alloca [36 x i8], align 8
  ; CHECK:	bl	__trampoline_setup
  call void @llvm.init.trampoline(ptr %tramp, ptr @f, ptr %nval)
  %fp = call ptr @llvm.adjust.trampoline(ptr %tramp)
  ret i64 0
}
