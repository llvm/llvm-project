; Make sure that the CGSCC pass manager can handle when instcombine simplifies
; one libcall into an unrelated libcall and update the call graph accordingly.
;
; RUN: opt -passes='cgscc(function(instcombine))' -S < %s | FileCheck %s

define i8* @wibble(i8* %arg1, i8* %arg2) {
; CHECK-LABLE: define @wibble(
bb:
  %tmp = alloca [1024 x i8], align 16
  %tmp2 = getelementptr inbounds [1024 x i8], [1024 x i8]* %tmp, i64 0, i64 0
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp2, i8* %arg1, i64 1024, i32 0, i1 false)
; CHECK:         call void @llvm.memcpy
  %tmp3 = call i64 @llvm.objectsize.i64.p0i8(i8* %tmp2, i1 false, i1 true)
  %tmp4 = call i8* @__strncpy_chk(i8* %arg2, i8* %tmp2, i64 1023, i64 %tmp3)
; CHECK-NOT:     call
; CHECK:         call i8* @strncpy(i8* %arg2, i8* %tmp2, i64 1023)
; CHECK-NOT:     call

  ret i8* %tmp4
; CHECK:         ret
}

define i8* @strncpy(i8* %arg1, i8* %arg2, i64 %size) {
bb:
  %result = call i8* @my_special_strncpy(i8* %arg1, i8* %arg2, i64 %size)
  ret i8* %result
}

declare i8* @my_special_strncpy(i8* %arg1, i8* %arg2, i64 %size)

declare i64 @llvm.objectsize.i64.p0i8(i8*, i1, i1)

declare i8* @__strncpy_chk(i8*, i8*, i64, i64)

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1)
