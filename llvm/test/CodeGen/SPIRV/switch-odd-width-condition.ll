; A switch on an odd-width integer type.

; RUN: llc -verify-machineinstrs -O2 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O2 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#I32:]] = OpTypeInt 32 0
; CHECK: %[[#COND:]] = OpFunctionParameter %[[#I32]]
; CHECK: OpSwitch %[[#COND]] %[[#]] 0 %[[#]] 1 %[[#]] 2 %[[#]]

define spir_func void @fuzz_kernel(ptr addrspace(1) %in, ptr addrspace(1) %out, i31 %n) {
entry:
  switch i31 %n, label %d [
    i31 0, label %m
    i31 1, label %b
    i31 2, label %c
  ]
b:
  br label %m
c:
  br label %m
d:
  br label %m
m:
  %r = phi i31 [ 40, %d ], [ 20, %b ], [ 30, %c ], [ 1, %entry ]
  store i31 %r, ptr addrspace(1) %out, align 4
  ret void
}

; The switch condition is also used by another instruction.

; CHECK: %[[#COND2:]] = OpFunctionParameter %[[#I32]]
; CHECK: OpSwitch %[[#COND2]] %[[#]] 0 %[[#]] 1 %[[#]] 2 %[[#]]

define spir_func void @cond_used_elsewhere(ptr addrspace(1) %out, i31 %n) {
entry:
  switch i31 %n, label %d [
    i31 0, label %a
    i31 1, label %b
    i31 2, label %c
  ]
a:
  br label %m
b:
  br label %m
c:
  br label %m
d:
  br label %m
m:
  %r = phi i31 [ 40, %d ], [ 20, %a ], [ 30, %b ], [ 50, %c ]
  %add = add i31 %r, %n
  store i31 %add, ptr addrspace(1) %out, align 4
  ret void
}
