; RUN: llc -mtriple=arm64-apple-ios %s -filetype=obj -o - | llvm-objdump --macho --section __DATA,__data - | FileCheck %s

; CHECK: Contents of (__DATA,__data) section
; CHECK: 0000002a 59ed145d
@other = global i32 42
@var = global i32 sub(i32 646102975,
                      i32 add (i32 trunc(i64 sub(i64 ptrtoint(ptr @var to i64),
                                                         i64 ptrtoint(ptr @other to i64)) to i32),
                               i32 3432360802))
