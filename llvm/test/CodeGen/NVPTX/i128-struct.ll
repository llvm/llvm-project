; RUN: llc < %s -O0 -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -O0 -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; CHECK-LABEL: .visible .func (.param .align 16 .b8 func_retval0[32]) foo(
define { i128, i128 } @foo(i64 %a, i32 %b) {
  %1 = sext i64 %a to i128
  %2 = sext i32 %b to i128
  %3 = insertvalue { i128, i128 } undef, i128 %1, 0
  %4 = insertvalue { i128, i128 } %3, i128 %2, 1

  ; CHECK: st.param.v2.b64 [func_retval0+0],  {%[[REG1:rd[0-9]+]], %[[REG2:rd[0-9]+]]};
  ; CHECK: st.param.v2.b64 [func_retval0+16], {%[[REG3:rd[0-9]+]], %[[REG4:rd[0-9]+]]};
  ret { i128, i128 } %4
}
