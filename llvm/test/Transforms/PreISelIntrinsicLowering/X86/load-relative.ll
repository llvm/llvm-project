; RUN: opt -mtriple=x86_64-pc-linux-gnu -pre-isel-intrinsic-lowering -S -o - %s | FileCheck %s

; CHECK: define ptr @foo32(ptr [[P:%.*]], i32 [[O:%.*]])
define ptr @foo32(ptr %p, i32 %o) {
  ; CHECK: [[OP:%.*]] = getelementptr i8, ptr [[P]], i32 [[O]]
  ; CHECK: [[OI32:%.*]] = load i32, ptr [[OP]], align 4
  ; CHECK: [[R:%.*]] = getelementptr i8, ptr [[P]], i32 [[OI32]]
  ; CHECK: ret ptr [[R]]
  %l = call ptr @llvm.load.relative.i32(ptr %p, i32 %o)
  ret ptr %l
}

; CHECK: define ptr @foo64(ptr [[P:%.*]], i64 [[O:%.*]])
define ptr @foo64(ptr %p, i64 %o) {
  ; CHECK: [[OP:%.*]] = getelementptr i8, ptr [[P]], i64 [[O]]
  ; CHECK: [[OI32:%.*]] = load i32, ptr [[OP]], align 4
  ; CHECK: [[R:%.*]] = getelementptr i8, ptr [[P]], i32 [[OI32]]
  ; CHECK: ret ptr [[R]]
  %l = call ptr @llvm.load.relative.i64(ptr %p, i64 %o)
  ret ptr %l
}

declare ptr @llvm.load.relative.i32(ptr, i32)
declare ptr @llvm.load.relative.i64(ptr, i64)
