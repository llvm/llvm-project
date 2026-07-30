; Test that we don't crash in the case where the type info object is an
; alias pointing to a non-zero offset within a global.

; RUN: llc < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

%struct.exception = type { i8, i8, i32, ptr, ptr, i32, ptr }
@g = global %struct.exception zeroinitializer
@a = alias %struct.exception, ptr inttoptr (i64 add (i64 ptrtoint (ptr @g to i64), i64 1297036692682702848) to ptr)

define i32 @typeid() {
  ; CHECK: movl $1, %eax
  %eh_typeid = tail call i32 @llvm.eh.typeid.for( ptr @a)
  ret i32 %eh_typeid
}

declare i32 @llvm.eh.typeid.for(ptr)
