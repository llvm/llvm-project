; RUN: not llc < %s -march=nvptx -mattr=+ptx72 -mcpu=sm_52 2>&1 | FileCheck %s --check-prefixes=CHECK-FAILS
; RUN: not llc < %s -march=nvptx -mattr=+ptx73 -mcpu=sm_50 2>&1 | FileCheck %s --check-prefixes=CHECK-FAILS

; RUN: llc < %s -march=nvptx -mattr=+ptx73 -mcpu=sm_52 | FileCheck %s --check-prefixes=CHECK,CHECK-32
; RUN: llc < %s -march=nvptx64 -mattr=+ptx73 -mcpu=sm_52 | FileCheck %s --check-prefixes=CHECK,CHECK-64
; RUN: %if ptxas %{ llc < %s -march=nvptx -mattr=+ptx73 -mcpu=sm_52 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mattr=+ptx73 -mcpu=sm_52 | %ptxas-verify %}

; CHECK-FAILS: in function test_dynamic_stackalloc{{.*}}: Support for dynamic alloca introduced in PTX ISA version 7.3 and requires target sm_52.

; CHECK-LABEL: .visible .func  (.param .b32 func_retval0) test_dynamic_stackalloc(
; CHECK-NOT: __local_depot

; CHECK-32:       ld.param.u32  %r[[SIZE:[0-9]]], [test_dynamic_stackalloc_param_0];
; CHECK-32-NEXT:  mad.lo.s32 %r[[SIZE2:[0-9]]], %r[[SIZE]], 1, 7;
; CHECK-32-NEXT:  and.b32         %r[[SIZE3:[0-9]]], %r[[SIZE2]], -8;
; CHECK-32-NEXT:  alloca.u32  %r[[ALLOCA:[0-9]]], %r[[SIZE3]], 16;
; CHECK-32-NEXT:  cvta.local.u32  %r[[ALLOCA]], %r[[ALLOCA]];
; CHECK-32-NEXT:  { // callseq 0, 0
; CHECK-32-NEXT:  .reg .b32 temp_param_reg;
; CHECK-32-NEXT:  .param .b32 param0;
; CHECK-32-NEXT:  st.param.b32  [param0+0], %r[[ALLOCA]];

; CHECK-64:       ld.param.u64  %rd[[SIZE:[0-9]]], [test_dynamic_stackalloc_param_0];
; CHECK-64-NEXT:  add.s64 %rd[[SIZE2:[0-9]]], %rd[[SIZE]], 7;
; CHECK-64-NEXT:  and.b64 %rd[[SIZE3:[0-9]]], %rd[[SIZE2]], -8;
; CHECK-64-NEXT:  alloca.u64  %rd[[ALLOCA:[0-9]]], %rd[[SIZE3]], 16;
; CHECK-64-NEXT:  cvta.local.u64  %rd[[ALLOCA]], %rd[[ALLOCA]];
; CHECK-64-NEXT:  { // callseq 0, 0
; CHECK-64-NEXT:  .reg .b32 temp_param_reg;
; CHECK-64-NEXT:  .param .b64 param0;
; CHECK-64-NEXT:  st.param.b64  [param0+0], %rd[[ALLOCA]];

; CHECK-NEXT:     .param .b32 retval0;
; CHECK-NEXT:     call.uni (retval0),
; CHECK-NEXT:     bar,

define i32 @test_dynamic_stackalloc(i64 %n) {
  %alloca = alloca i8, i64 %n, align 16
  %call = call i32 @bar(ptr %alloca)
  ret i32 %call
}

declare i32 @bar(ptr)