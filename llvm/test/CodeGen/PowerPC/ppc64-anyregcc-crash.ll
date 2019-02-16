; RUN: not llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu 2>&1 | FileCheck %s
;
; Check that misuse of anyregcc results in a compile time error.

; CHECK: LLVM ERROR: ran out of registers during register allocation
define i64 @anyreglimit(i64 %v1, i64 %v2, i64 %v3, i64 %v4, i64 %v5, i64 %v6, i64 %v7, i64 %v8,
                        i64 %v9, i64 %v10, i64 %v11, i64 %v12, i64 %v13, i64 %v14, i64 %v15, i64 %v16,
                        i64 %v17, i64 %v18, i64 %v19, i64 %v20, i64 %v21, i64 %v22, i64 %v23, i64 %v24,
                        i64 %v25, i64 %v26, i64 %v27, i64 %v28, i64 %v29, i64 %v30, i64 %v31, i64 %v32) {
entry:
  %result = tail call anyregcc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 12, i32 15, i8* inttoptr (i64 0 to i8*), i32 32,
                i64 %v1, i64 %v2, i64 %v3, i64 %v4, i64 %v5, i64 %v6, i64 %v7, i64 %v8,
                i64 %v9, i64 %v10, i64 %v11, i64 %v12, i64 %v13, i64 %v14, i64 %v15, i64 %v16,
                i64 %v17, i64 %v18, i64 %v19, i64 %v20, i64 %v21, i64 %v22, i64 %v23, i64 %v24,
                i64 %v25, i64 %v26, i64 %v27, i64 %v28, i64 %v29, i64 %v30, i64 %v31, i64 %v32)
  ret i64 %result
}

declare i64 @llvm.experimental.patchpoint.i64(i64, i32, i8*, i32, ...)
