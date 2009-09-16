; RUN: opt < %s -S -memcpyopt | FileCheck %s

; The resulting memset is only 4-byte aligned, despite containing
; a 16-byte alignmed store in the middle.

; CHECK: call void @llvm.memset.i64(i8* %a01, i8 0, i64 16, i32 4)

define void @foo(i32* %p) {
  %a0 = getelementptr i32* %p, i64 0
  store i32 0, i32* %a0, align 4
  %a1 = getelementptr i32* %p, i64 1
  store i32 0, i32* %a1, align 16
  %a2 = getelementptr i32* %p, i64 2
  store i32 0, i32* %a2, align 4
  %a3 = getelementptr i32* %p, i64 3
  store i32 0, i32* %a3, align 4
  ret void
}
