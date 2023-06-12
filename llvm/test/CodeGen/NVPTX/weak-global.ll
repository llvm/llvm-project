; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; CHECK: .weak .global .align 4 .u32 g
@g = common addrspace(1) global i32 zeroinitializer

define i32 @func0() {
  %val = load i32, ptr addrspace(1) @g
  ret i32 %val
}
