; The goal of the test is to ensure that type inference doesn't break validity of the generated SPIR-V code.
; The only pass criterion is that spirv-val considers output valid.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpFunction

%class.anon = type { i32, ptr addrspace(4)}

define weak_odr dso_local spir_kernel void @foo(i32 noundef %_arg_N, i1 %fl) {
entry:
  %__SYCLKernel = alloca %class.anon, align 8
  store i32 %_arg_N, ptr %__SYCLKernel, align 8
  br label %arinit

arinit:
  %scevgep3 = getelementptr nuw i8, ptr %__SYCLKernel, i64 24
  br label %for.cond.i

for.cond.i:
  %lsr.iv4 = phi ptr [ %scevgep5, %for.body.i ], [ %scevgep3, %arinit ]
  br i1 %fl, label %for.body.i, label %exit

for.body.i:
  %scevgep6 = getelementptr i8, ptr %lsr.iv4, i64 -8
  %_M_value.imag.i.i = load double, ptr %lsr.iv4, align 8
  %scevgep5 = getelementptr i8, ptr %lsr.iv4, i64 32
  br label %for.cond.i

exit:
  ret void
}
