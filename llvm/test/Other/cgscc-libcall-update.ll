; Make sure that the CGSCC pass manager can handle when instcombine simplifies
; one libcall into an unrelated libcall and update the call graph accordingly.
;
; Also check that it can handle inlining *removing* a libcall entirely.
;
; Finally, we include some recursive patterns and forced analysis invaliadtion
; that can trigger infinite CGSCC refinement if not handled correctly.
;
; RUN: opt -passes='cgscc(inline,function(instcombine,invalidate<all>))' -S < %s | FileCheck %s

define ptr @wibble(ptr %arg1, ptr %arg2) {
; CHECK-LABEL: define ptr @wibble(
bb:
  %tmp = alloca [1024 x i8], align 16
  call void @llvm.memcpy.p0.p0.i64(ptr %tmp, ptr %arg1, i64 1024, i1 false)
; CHECK:         call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 16 dereferenceable(1024) 
  %tmp3 = call i64 @llvm.objectsize.i64.p0(ptr %tmp, i1 false, i1 true, i1 false)
  %tmp4 = call ptr @__strncpy_chk(ptr %arg2, ptr %tmp, i64 1023, i64 %tmp3)
; CHECK-NOT:     call
; CHECK:         call ptr @strncpy(ptr noundef nonnull dereferenceable(1) %arg2, ptr noundef nonnull dereferenceable(1) %tmp, i64 1023)

; CHECK-NOT:     call

  ret ptr %tmp4
; CHECK:         ret
}

define ptr @strncpy(ptr %arg1, ptr %arg2, i64 %size) noinline {
bb:
; CHECK:         call ptr @my_special_strncpy(ptr %arg1, ptr %arg2, i64 %size)
  %result = call ptr @my_special_strncpy(ptr %arg1, ptr %arg2, i64 %size)
  ret ptr %result
}

declare ptr @my_special_strncpy(ptr %arg1, ptr %arg2, i64 %size)

declare i64 @llvm.objectsize.i64.p0(ptr, i1, i1, i1)

declare ptr @__strncpy_chk(ptr, ptr, i64, i64)

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1)

; Check that even when we completely remove a libcall we don't get the call
; graph wrong once we handle libcalls in the call graph specially to address
; the above case.
define i32 @hoge(ptr %arg1) {
; CHECK-LABEL: define i32 @hoge(
bb:
  %tmp41 = load ptr, ptr null
  %tmp6 = load i32, ptr %arg1
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

define i64 @write(i32 %i, ptr %p, i64 %j) {
entry:
  %val = call i64 @write_wrapper(i32 %i, ptr %p, i64 %j) noinline
  ret i64 %val
}

define i64 @write_wrapper(i32 %i, ptr %p, i64 %j) {
entry:
  %val = call i64 @write(i32 %i, ptr %p, i64 %j) noinline
  ret i64 %val
}
