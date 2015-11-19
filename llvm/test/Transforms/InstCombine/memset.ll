; RUN: opt < %s -instcombine -S | not grep "call.*llvm.memset"

define i32 @main() {
  %target = alloca [1024 x i8]
  %target_p = getelementptr [1024 x i8], [1024 x i8]* %target, i32 0, i32 0
  call void @llvm.memset.p0i8.i32(i8* %target_p, i8 1, i32 0, i32 1, i1 false)
  call void @llvm.memset.p0i8.i32(i8* %target_p, i8 1, i32 1, i32 1, i1 false)
  call void @llvm.memset.p0i8.i32(i8* %target_p, i8 1, i32 2, i32 2, i1 false)
  call void @llvm.memset.p0i8.i32(i8* %target_p, i8 1, i32 4, i32 4, i1 false)
  call void @llvm.memset.p0i8.i32(i8* %target_p, i8 1, i32 8, i32 8, i1 false)
  ret i32 0
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind
