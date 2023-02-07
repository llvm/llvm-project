; Check that various LLVM idioms get lowered to NVPTX as expected.

; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

%struct.S16 = type { i16, i16 }
%struct.S32 = type { i32, i32 }

; CHECK-LABEL: abs_i16(
define i16 @abs_i16(i16 %a) {
; CHECK: abs.s16
  %neg = sub i16 0, %a
  %abs.cond = icmp sge i16 %a, 0
  %abs = select i1 %abs.cond, i16 %a, i16 %neg
  ret i16 %abs
}

; CHECK-LABEL: abs_i32(
define i32 @abs_i32(i32 %a) {
; CHECK: abs.s32
  %neg = sub i32 0, %a
  %abs.cond = icmp sge i32 %a, 0
  %abs = select i1 %abs.cond, i32 %a, i32 %neg
  ret i32 %abs
}

; CHECK-LABEL: abs_i64(
define i64 @abs_i64(i64 %a) {
; CHECK: abs.s64
  %neg = sub i64 0, %a
  %abs.cond = icmp sge i64 %a, 0
  %abs = select i1 %abs.cond, i64 %a, i64 %neg
  ret i64 %abs
}

; CHECK-LABEL: i32_to_2xi16(
define %struct.S16 @i32_to_2xi16(i32 noundef %in) {
  %low = trunc i32 %in to i16
  %high32 = lshr i32 %in, 16
  %high = trunc i32 %high32 to i16
; CHECK:       ld.param.u32  %[[R32:r[0-9]+]], [i32_to_2xi16_param_0];
; CHECK-DAG:   cvt.u16.u32   %rs{{[0-9+]}}, %[[R32]];
; CHECK-DAG    mov.b32       {tmp, %rs{{[0-9+]}}}, %[[R32]];
  %s1 = insertvalue %struct.S16 poison, i16 %low, 0
  %s = insertvalue %struct.S16 %s1, i16 %high, 1
  ret %struct.S16 %s
}

; CHECK-LABEL: i32_to_2xi16_lh(
; Same as above, but with rearranged order of low/high parts.
define %struct.S16 @i32_to_2xi16_lh(i32 noundef %in) {
  %high32 = lshr i32 %in, 16
  %high = trunc i32 %high32 to i16
  %low = trunc i32 %in to i16
; CHECK:       ld.param.u32  %[[R32:r[0-9]+]], [i32_to_2xi16_lh_param_0];
; CHECK-DAG:   cvt.u16.u32   %rs{{[0-9+]}}, %[[R32]];
; CHECK-DAG    mov.b32       {tmp, %rs{{[0-9+]}}}, %[[R32]];
  %s1 = insertvalue %struct.S16 poison, i16 %low, 0
  %s = insertvalue %struct.S16 %s1, i16 %high, 1
  ret %struct.S16 %s
}


; CHECK-LABEL: i32_to_2xi16_not(
define %struct.S16 @i32_to_2xi16_not(i32 noundef %in) {
  %low = trunc i32 %in to i16
  ;  Shift by any value other than 16 blocks the conversiopn to mov.
  %high32 = lshr i32 %in, 15
  %high = trunc i32 %high32 to i16
; CHECK:       cvt.u16.u32
; CHECK:       shr.u32
; CHECK:       cvt.u16.u32
  %s1 = insertvalue %struct.S16 poison, i16 %low, 0
  %s = insertvalue %struct.S16 %s1, i16 %high, 1
  ret %struct.S16 %s
}

; CHECK-LABEL: i64_to_2xi32(
define %struct.S32 @i64_to_2xi32(i64 noundef %in) {
  %low = trunc i64 %in to i32
  %high64 = lshr i64 %in, 32
  %high = trunc i64 %high64 to i32
; CHECK:       ld.param.u64  %[[R64:rd[0-9]+]], [i64_to_2xi32_param_0];
; CHECK-DAG:   cvt.u32.u64   %r{{[0-9+]}}, %[[R64]];
; CHECK-DAG    mov.b64       {tmp, %r{{[0-9+]}}}, %[[R64]];
  %s1 = insertvalue %struct.S32 poison, i32 %low, 0
  %s = insertvalue %struct.S32 %s1, i32 %high, 1
  ret %struct.S32 %s
}

; CHECK-LABEL: i64_to_2xi32_not(
define %struct.S32 @i64_to_2xi32_not(i64 noundef %in) {
  %low = trunc i64 %in to i32
  ;  Shift by any value other than 32 blocks the conversiopn to mov.
  %high64 = lshr i64 %in, 31
  %high = trunc i64 %high64 to i32
; CHECK:       cvt.u32.u64
; CHECK:       shr.u64
; CHECK:       cvt.u32.u64
  %s1 = insertvalue %struct.S32 poison, i32 %low, 0
  %s = insertvalue %struct.S32 %s1, i32 %high, 1
  ret %struct.S32 %s
}

; CHECK-LABEL: i32_to_2xi16_shr(
; Make sure we do not get confused when our input itself is [al]shr.
define %struct.S16 @i32_to_2xi16_shr(i32 noundef %i){
  call void @escape_int(i32 %i); // Force %i to be loaded completely.
  %i1 = ashr i32 %i, 16
  %l = trunc i32 %i1 to i16
  %h32 = ashr i32 %i1, 16
  %h = trunc i32 %h32 to i16
; CHECK:      ld.param.u32    %[[R32:r[0-9]+]], [i32_to_2xi16_shr_param_0];
; CHECK:      shr.s32         %[[R32H:r[0-9]+]], %[[R32]], 16;
; CHECK-DAG    mov.b32       {tmp, %rs{{[0-9+]}}}, %[[R32]];
; CHECK-DAG    mov.b32       {tmp, %rs{{[0-9+]}}}, %[[R32H]];
  %s0 = insertvalue %struct.S16 poison, i16 %l, 0
  %s1 = insertvalue %struct.S16 %s0, i16 %h, 1
  ret %struct.S16 %s1
}
declare dso_local void @escape_int(i32 noundef)

