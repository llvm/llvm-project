; RUN: not llc < %s -mtriple=x86_64-apple-darwin 2>&1 | FileCheck %s
;
; Check that misuse of anyregcc results in a compile time error.

; CHECK: error: ran out of registers during register allocation
define i64 @anyreglimit(i64 %v1, i64 %v2, i64 %v3, i64 %v4, i64 %v5, i64 %v6,
                        i64 %v7, i64 %v8, i64 %v9, i64 %v10, i64 %v11, i64 %v12,
                        i64 %v13, i64 %v14, i64 %v15, i64 %v16) {
entry:
  %result = tail call anyregcc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 12, i32 15, ptr inttoptr (i64 0 to ptr), i32 16,
                i64 %v1, i64 %v2, i64 %v3, i64 %v4, i64 %v5, i64 %v6,
                i64 %v7, i64 %v8, i64 %v9, i64 %v10, i64 %v11, i64 %v12,
                i64 %v13, i64 %v14, i64 %v15, i64 %v16)
  ret i64 %result
}

declare i64 @llvm.experimental.patchpoint.i64(i64, i32, ptr, i32, ...)
