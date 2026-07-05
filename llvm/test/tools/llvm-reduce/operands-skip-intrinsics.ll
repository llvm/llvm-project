; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-skip --test FileCheck --test-arg --check-prefixes=INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK %s < %t

; Make sure address captures of intrinsics are not introduced.


declare ptr @fptr()
declare ptr @fptr.keep()
declare ptr @llvm.intrin()
declare ptr @llvm.intrin.keep()
declare void @func(i32, ptr)

declare ptr @llvm.intrin2()
declare ptr @llvm.intrin.chain(ptr)

; INTERESTING-LABEL: define void @caller(
; INTERESTING: call void @func(i32 1
; INTERESTING: call void @func(i32 3
; INTERESTING: call void @func(i32 4
; INTERESTING: call void @func(i32 5

; CHECK: %intrin.ptr.keep = call ptr @llvm.intrin.keep()

; CHECK: call void @func(i32 0, ptr @fptr)
; CHECK: call void @func(i32 1, ptr @fptr.keep)
; CHECK: call void @func(i32 2, ptr %intrin.ptr)
; CHECK: call void @func(i32 3, ptr %intrin.ptr.keep)
; CHECK: call void @func(i32 4, ptr %intrin.ptr.keep)
; CHECK: call void @func(i32 5, ptr %chained.ptr)
define void @caller() {
  %func.ptr = call ptr @fptr()
  %func.ptr.keep = call ptr @fptr.keep()
  %intrin.ptr = call ptr @llvm.intrin()
  %intrin.ptr.keep = call ptr @llvm.intrin.keep()
  %gep = getelementptr i8, ptr %intrin.ptr.keep, i64 128
  %chained.ptr = call ptr @llvm.intrin2()
  %chain.ptr = call ptr @llvm.intrin.chain(ptr %chained.ptr)
  call void @func(i32 0, ptr %func.ptr)
  call void @func(i32 1, ptr %func.ptr.keep)
  call void @func(i32 2, ptr %intrin.ptr)
  call void @func(i32 3, ptr %intrin.ptr.keep)
  call void @func(i32 4, ptr %gep)
  call void @func(i32 5, ptr %chain.ptr)
  ret void
}
