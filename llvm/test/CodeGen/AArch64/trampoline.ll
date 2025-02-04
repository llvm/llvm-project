; RUN: llc -mtriple=aarch64-- < %s | FileCheck %s

@trampg = internal global [36 x i8] zeroinitializer, align 8

declare void @llvm.init.trampoline(ptr, ptr, ptr);
declare ptr @llvm.adjust.trampoline(ptr);

define i64 @f(ptr nest %c, i64 %x, i64 %y) {
  %sum = add i64 %x, %y
  ret i64 %sum
}

define i64 @func1() {
  %val = alloca i64
  %nval = bitcast ptr %val to ptr
  %tramp = alloca [36 x i8], align 8
  ; CHECK:	mov	w1, #36
  ; CHECK:	bl	__trampoline_setup
  call void @llvm.init.trampoline(ptr %tramp, ptr @f, ptr %nval)
  %fp = call ptr @llvm.adjust.trampoline(ptr %tramp)
  ret i64 0
}

define i64 @func2() {
  %val = alloca i64
  %nval = bitcast ptr %val to ptr
  ; CHECK:	mov	w1, #36
  ; CHECK:	bl	__trampoline_setup
  call void @llvm.init.trampoline(ptr @trampg, ptr @f, ptr %nval)
  %fp = call ptr @llvm.adjust.trampoline(ptr @trampg)
  ret i64 0
}
