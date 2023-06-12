; RUN: llc -O2 < %s -march=nvptx64 -mcpu=sm_35 | FileCheck %s --check-prefix=O2 --check-prefix=CHECK
; RUN: llc -O0 < %s -march=nvptx64 -mcpu=sm_35 | FileCheck %s --check-prefix=O0 --check-prefix=CHECK
; RUN: %if ptxas %{ llc -O2 < %s -march=nvptx64 -mcpu=sm_35 | %ptxas-verify %}
; RUN: %if ptxas %{ llc -O0 < %s -march=nvptx64 -mcpu=sm_35 | %ptxas-verify %}

; The following IR
;
;   quot = n / d
;   rem  = n % d
;
; should be transformed into
;
;   quot = n / d
;   rem = n - (n / d) * d
;
; during NVPTX isel, at -O2.  At -O0, we should leave it alone.

; CHECK-LABEL: sdiv32(
define void @sdiv32(i32 %n, i32 %d, ptr %quot_ret, ptr %rem_ret) {
  ; CHECK: div.s32 [[quot:%r[0-9]+]], [[num:%r[0-9]+]], [[den:%r[0-9]+]];
  %quot = sdiv i32 %n, %d

  ; O0: rem.s32
  ; (This is unfortunately order-sensitive, even though mul is commutative.)
  ; O2: mul.lo.s32 [[mul:%r[0-9]+]], [[quot]], [[den]];
  ; O2: sub.s32 [[rem:%r[0-9]+]], [[num]], [[mul]]
  %rem = srem i32 %n, %d

  ; O2: st{{.*}}[[quot]]
  store i32 %quot, ptr %quot_ret
  ; O2: st{{.*}}[[rem]]
  store i32 %rem, ptr %rem_ret
  ret void
}

; CHECK-LABEL: udiv32(
define void @udiv32(i32 %n, i32 %d, ptr %quot_ret, ptr %rem_ret) {
  ; CHECK: div.u32 [[quot:%r[0-9]+]], [[num:%r[0-9]+]], [[den:%r[0-9]+]];
  %quot = udiv i32 %n, %d

  ; O0: rem.u32

  ; Selection DAG doesn't know whether this is signed or unsigned
  ; multiplication and subtraction, but it doesn't make a difference either
  ; way.
  ; O2: mul.lo.{{u|s}}32 [[mul:%r[0-9]+]], [[quot]], [[den]];
  ; O2: sub.{{u|s}}32 [[rem:%r[0-9]+]], [[num]], [[mul]]
  %rem = urem i32 %n, %d

  ; O2: st{{.*}}[[quot]]
  store i32 %quot, ptr %quot_ret
  ; O2: st{{.*}}[[rem]]
  store i32 %rem, ptr %rem_ret
  ret void
}

; Check that we don't perform this optimization if one operation is signed and
; the other isn't.
; CHECK-LABEL: mismatched_types1(
define void @mismatched_types1(i32 %n, i32 %d, ptr %quot_ret, ptr %rem_ret) {
  ; CHECK: div.u32
  ; CHECK: rem.s32
  %quot = udiv i32 %n, %d
  %rem = srem i32 %n, %d
  store i32 %quot, ptr %quot_ret
  store i32 %rem, ptr %rem_ret
  ret void
}

; CHECK-LABEL: mismatched_types2(
define void @mismatched_types2(i32 %n, i32 %d, ptr %quot_ret, ptr %rem_ret) {
  ; CHECK: div.s32
  ; CHECK: rem.u32
  %quot = sdiv i32 %n, %d
  %rem = urem i32 %n, %d
  store i32 %quot, ptr %quot_ret
  store i32 %rem, ptr %rem_ret
  ret void
}

; Check that we don't perform this optimization if the inputs to the div don't
; match the inputs to the rem.
; CHECK-LABEL: mismatched_inputs1(
define void @mismatched_inputs1(i32 %n, i32 %d, ptr %quot_ret, ptr %rem_ret) {
  ; CHECK: div.s32
  ; CHECK: rem.s32
  %quot = sdiv i32 %n, %d
  %rem = srem i32 %d, %n
  store i32 %quot, ptr %quot_ret
  store i32 %rem, ptr %rem_ret
  ret void
}

; CHECK-LABEL: mismatched_inputs2(
define void @mismatched_inputs2(i32 %n1, i32 %n2, i32 %d, ptr %quot_ret, ptr %rem_ret) {
  ; CHECK: div.s32
  ; CHECK: rem.s32
  %quot = sdiv i32 %n1, %d
  %rem = srem i32 %n2, %d
  store i32 %quot, ptr %quot_ret
  store i32 %rem, ptr %rem_ret
  ret void
}

; CHECK-LABEL: mismatched_inputs3(
define void @mismatched_inputs3(i32 %n, i32 %d1, i32 %d2, ptr %quot_ret, ptr %rem_ret) {
  ; CHECK: div.s32
  ; CHECK: rem.s32
  %quot = sdiv i32 %n, %d1
  %rem = srem i32 %n, %d2
  store i32 %quot, ptr %quot_ret
  store i32 %rem, ptr %rem_ret
  ret void
}
