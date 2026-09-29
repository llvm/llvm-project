; RUN: not --crash llc -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: data symbols must live in a data section: data_symbol

target triple = "wasm32-unknown-unknown"

@data_symbol = constant [1024 x i32] zeroinitializer, section ".text", align 16

define hidden i32 @main() local_unnamed_addr #0 {
entry:
  %0 = load i32, ptr getelementptr inbounds ([1024 x i32], ptr @data_symbol, i32 0, i32 10)
  ret i32 %0
}
