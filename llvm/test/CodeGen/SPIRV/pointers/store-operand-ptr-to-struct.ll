; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; TODO: OpFunctionParameter should be a pointer of struct base type.
; XFAIL: *

%struct = type {
  i32,
  i16
}

%nested_struct = type {
  %struct,
  i16
}

define void @foo(ptr %ptr) {
  store %nested_struct undef, ptr %ptr
  ret void
}
