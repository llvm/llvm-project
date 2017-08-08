; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @test([1024 x i8]* %target) {
  ;; CHECK-LABEL: test
  ;; CHECK-NEXT: [[P1:%[^\s]+]] = getelementptr inbounds [1024 x i8], [1024 x i8]* %target, i64 0, i64 0
  ;; CHECK-NEXT: store i8 1, i8* [[P1]], align 1
  ;; CHECK-NEXT: [[P2:%[^\s]+]] = bitcast [1024 x i8]* %target to i16*
  ;; CHECK-NEXT: store i16 257, i16* [[P2]], align 2
  ;; CHECK-NEXT: [[P3:%[^\s]+]] = bitcast [1024 x i8]* %target to i32*
  ;; CHECK-NEXT: store i32 16843009, i32* [[P3]], align 4
  ;; CHECK-NEXT: [[P4:%[^\s]+]] = bitcast [1024 x i8]* %target to i64*
  ;; CHECK-NEXT: store i64 72340172838076673, i64* [[P4]], align 8
  ;; CHECK-NEXT: ret i32 0
  %target_p = getelementptr [1024 x i8], [1024 x i8]* %target, i32 0, i32 0
  call void @llvm.memset.p0i8.i32(i8* %target_p, i8 1, i32 0, i32 1, i1 false)
  call void @llvm.memset.p0i8.i32(i8* %target_p, i8 1, i32 1, i32 1, i1 false)
  call void @llvm.memset.p0i8.i32(i8* %target_p, i8 1, i32 2, i32 2, i1 false)
  call void @llvm.memset.p0i8.i32(i8* %target_p, i8 1, i32 4, i32 4, i1 false)
  call void @llvm.memset.p0i8.i32(i8* %target_p, i8 1, i32 8, i32 8, i1 false)
  ret i32 0
}

declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i32, i1) argmemonly nounwind
