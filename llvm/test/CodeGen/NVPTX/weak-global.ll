; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 -mattr=+ptx43 | FileCheck %s --check-prefix PTX43
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 -mattr=+ptx50 | FileCheck %s --check-prefix PTX50
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 -mattr=+ptx43 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 -mattr=+ptx50 | %ptxas-verify %}

; PTX43: .weak .global .align 4 .u32 g
; PTX50: .common .global .align 4 .u32 g
@g = common addrspace(1) global i32 zeroinitializer

define i32 @func0() {
  %val = load i32, ptr addrspace(1) @g
  ret i32 %val
}
