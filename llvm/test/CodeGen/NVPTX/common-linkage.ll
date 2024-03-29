; RUN: llc < %s -march=nvptx -mcpu=sm_20 -mattr=+ptx43 | FileCheck %s --check-prefixes CHECK,PTX43
; RUN: llc < %s -march=nvptx -mcpu=sm_20 -mattr=+ptx50 | FileCheck %s --check-prefixes CHECK,PTX50
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 -mattr=+ptx43 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 -mattr=+ptx50 | %ptxas-verify %}

; PTX43: .weak .global .align 4 .u32 g
; PTX50: .common .global .align 4 .u32 g
@g = common addrspace(1) global i32 0, align 4

; CHECK: .weak .const .align 4 .u32 c
@c = common addrspace(4) global i32 0, align 4

; CHECK: .weak .shared .align 4 .u32 s
@s = common addrspace(3) global i32 0, align 4

define i32 @f1() {
  %1 = load i32, ptr addrspace(1) @g
  ret i32 %1
}

define i32 @f4() {
  %1 = load i32, ptr addrspace(4) @c
  ret i32 %1
}

define i32 @f3() {
  %1 = load i32, ptr addrspace(3) @s
  ret i32 %1
}
