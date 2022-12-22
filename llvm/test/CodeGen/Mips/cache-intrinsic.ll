; RUN: llc %s -o - | FileCheck %s
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"
target triple = "mips--linux-gnu"

@buffer = global [32 x i8] c"This is a largely unused buffer\00", align 1
@.str = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1
@.str1 = private unnamed_addr constant [25 x i8] c"Still, largely unused...\00", align 1

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval
  %call = call i32 (ptr, ...) @printf(ptr @.str, ptr @buffer)
  %call1 = call ptr @strcpy(ptr @buffer, ptr @.str1) #3
  call void @llvm.clear_cache(ptr @buffer, ptr getelementptr inbounds (i8, ptr @buffer, i32 32)) #3
  %call2 = call i32 (ptr, ...) @printf(ptr @.str, ptr @buffer)
  ret i32 0
}

; CHECK: __clear_cache

declare i32 @printf(ptr, ...)

declare ptr @strcpy(ptr, ptr)

declare void @llvm.clear_cache(ptr, ptr)
