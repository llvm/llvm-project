; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

; module asm must come after PTX version/target directives.
; CHECK-NOT: .global .b32 val;

; CHECK-DAG: .version
; CHECK-DAG: .target

; CHECK: .global .b32 val;
module asm ".global .b32 val;"

; module asm must happen before we emit other things.
; CHECK-LABEL: .visible .func foo
define void @foo() {
  ret void
}
; Make sure it does not show up anywhere else in the output.
; CHECK-NOT: .global .b32 val;
