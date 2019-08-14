; Make sure that the CGSCC pass manager can handle when instcombine simplifies
; one libcall into an unrelated libcall and update the call graph accordingly.
;
; Also check that it can handle inlining *removing* a libcall entirely.
;
; Finally, we include some recursive patterns and forced analysis invaliadtion
; that can trigger infinite CGSCC refinement if not handled correctly.
;
; RUN: opt -passes='cgscc(inline,function(instcombine,invalidate<all>))' -S < %s | FileCheck %s

define i8* @wibble(i8* %arg1, i8* %arg2) {
; CHECK-LABEL: define i8* @wibble(
bb:
  %tmp = alloca [1024 x i8], align 16
  %tmp2 = getelementptr inbounds [1024 x i8], [1024 x i8]* %tmp, i64 0, i64 0
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp2, i8* %arg1, i64 1024, i1 false)
; CHECK:         call void @llvm.memcpy
  %tmp3 = call i64 @llvm.objectsize.i64.p0i8(i8* %tmp2, i1 false, i1 true, i1 false)
  %tmp4 = call i8* @__strncpy_chk(i8* %arg2, i8* %tmp2, i64 1023, i64 %tmp3)
; CHECK-NOT:     call
; CHECK:         call i8* @strncpy(i8* %arg2, i8* nonnull %tmp2, i64 1023)
; CHECK-NOT:     call

  ret i8* %tmp4
; CHECK:         ret
}

define i8* @strncpy(i8* %arg1, i8* %arg2, i64 %size) noinline {
bb:
  %result = call i8* @my_special_strncpy(i8* %arg1, i8* %arg2, i64 %size)
  ret i8* %result
}

declare i8* @my_special_strncpy(i8* %arg1, i8* %arg2, i64 %size)

declare i64 @llvm.objectsize.i64.p0i8(i8*, i1, i1, i1)

declare i8* @__strncpy_chk(i8*, i8*, i64, i64)

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

; Check that even when we completely remove a libcall we don't get the call
; graph wrong once we handle libcalls in the call graph specially to address
; the above case.
define i32 @hoge(i32* %arg1) {
; CHECK-LABEL: define i32 @hoge(
bb:
  %tmp41 = load i32*, i32** null
  %tmp6 = load i32, i32* %arg1
  %tmp7 = call i32 @ntohl(i32 %tmp6)
; CHECK-NOT: call i32 @ntohl
  ret i32 %tmp7
; CHECK: ret i32
}

; Even though this function is not used, it should be retained as it may be
; used when doing further libcall transformations.
define internal i32 @ntohl(i32 %x) {
; CHECK-LABEL: define internal i32 @ntohl(
entry:
  %and2 = lshr i32 %x, 8
  %shr = and i32 %and2, 65280
  ret i32 %shr
}

define i64 @write(i32 %i, i8* %p, i64 %j) {
entry:
  %val = call i64 @write_wrapper(i32 %i, i8* %p, i64 %j) noinline
  ret i64 %val
}

define i64 @write_wrapper(i32 %i, i8* %p, i64 %j) {
entry:
  %val = call i64 @write(i32 %i, i8* %p, i64 %j) noinline
  ret i64 %val
}
