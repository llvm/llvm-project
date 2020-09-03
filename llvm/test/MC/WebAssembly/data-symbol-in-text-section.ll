; RUN: not --crash llc -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: data symbols must live in a data section: data_symbol

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

@data_symbol = constant [1024 x i32] zeroinitializer, section ".text", align 16

define hidden i32 @main() local_unnamed_addr #0 {
entry:
  %0 = load i32, i32* getelementptr inbounds ([1024 x i32], [1024 x i32]* @data_symbol, i32 0, i32 10)
  ret i32 %0
}
