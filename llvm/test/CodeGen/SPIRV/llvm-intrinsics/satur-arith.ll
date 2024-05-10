; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] u_add_sat
; CHECK-NEXT: %[[#]] = OpExtInst %[[#]] %[[#]] u_sub_sat
; CHECK-NEXT: %[[#]] = OpExtInst %[[#]] %[[#]] s_add_sat
; CHECK-NEXT: %[[#]] = OpExtInst %[[#]] %[[#]] s_sub_sat

define spir_func void @foo(i16 %x, i16 %y) {
entry:
  %r1 = tail call i16 @llvm.uadd.sat.i16(i16 %x, i16 %y)
  %r2 = tail call i16 @llvm.usub.sat.i16(i16 %x, i16 %y)
  %r3 = tail call i16 @llvm.sadd.sat.i16(i16 %x, i16 %y)
  %r4 = tail call i16 @llvm.ssub.sat.i16(i16 %x, i16 %y)
  ret void
}
