; RUN: opt -pre-isel-intrinsic-lowering -S -o - %s | FileCheck %s

; CHECK: define i8* @foo32(i8* [[P:%.*]], i32 [[O:%.*]])
define i8* @foo32(i8* %p, i32 %o) {
  ; CHECK: [[OP:%.*]] = getelementptr i8, i8* [[P]], i32 [[O]]
  ; CHECK: [[OPI32:%.*]] = bitcast i8* [[OP]] to i32*
  ; CHECK: [[OI32:%.*]] = load i32, i32* [[OPI32]], align 4
  ; CHECK: [[R:%.*]] = getelementptr i8, i8* [[P]], i32 [[OI32]]
  ; CHECK: ret i8* [[R]]
  %l = call i8* @llvm.load.relative.i32(i8* %p, i32 %o)
  ret i8* %l
}

; CHECK: define i8* @foo64(i8* [[P:%.*]], i64 [[O:%.*]])
define i8* @foo64(i8* %p, i64 %o) {
  ; CHECK: [[OP:%.*]] = getelementptr i8, i8* [[P]], i64 [[O]]
  ; CHECK: [[OPI32:%.*]] = bitcast i8* [[OP]] to i32*
  ; CHECK: [[OI32:%.*]] = load i32, i32* [[OPI32]], align 4
  ; CHECK: [[R:%.*]] = getelementptr i8, i8* [[P]], i32 [[OI32]]
  ; CHECK: ret i8* [[R]]
  %l = call i8* @llvm.load.relative.i64(i8* %p, i64 %o)
  ret i8* %l
}

declare i8* @llvm.load.relative.i32(i8*, i32)
declare i8* @llvm.load.relative.i64(i8*, i64)
