; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

declare i16 @llvm.ctlz.i16(i16, i1) readnone
declare i32 @llvm.ctlz.i32(i32, i1) readnone
declare i64 @llvm.ctlz.i64(i64, i1) readnone

; There should be no difference between llvm.ctlz.i32(%a, true) and
; llvm.ctlz.i32(%a, false), as ptx's clz(0) is defined to return 0.

; CHECK-LABEL: myctlz(
define i32 @myctlz(i32 %a) {
; CHECK: ld.param.
; CHECK-NEXT: clz.b32
; CHECK-NEXT: st.param.
; CHECK-NEXT: ret;
  %val = call i32 @llvm.ctlz.i32(i32 %a, i1 false) readnone
  ret i32 %val
}
; CHECK-LABEL: myctlz_2(
define i32 @myctlz_2(i32 %a) {
; CHECK: ld.param.
; CHECK-NEXT: clz.b32
; CHECK-NEXT: st.param.
; CHECK-NEXT: ret;
  %val = call i32 @llvm.ctlz.i32(i32 %a, i1 true) readnone
  ret i32 %val
}

; PTX's clz.b64 returns a 32-bit value, but LLVM's intrinsic returns a 64-bit
; value, so here we have to zero-extend it.
; CHECK-LABEL: myctlz64(
define i64 @myctlz64(i64 %a) {
; CHECK: ld.param.
; CHECK-NEXT: clz.b64
; CHECK-NEXT: cvt.u64.u32
; CHECK-NEXT: st.param.
; CHECK-NEXT: ret;
  %val = call i64 @llvm.ctlz.i64(i64 %a, i1 false) readnone
  ret i64 %val
}
; CHECK-LABEL: myctlz64_2(
define i64 @myctlz64_2(i64 %a) {
; CHECK: ld.param.
; CHECK-NEXT: clz.b64
; CHECK-NEXT: cvt.u64.u32
; CHECK-NEXT: st.param.
; CHECK-NEXT: ret;
  %val = call i64 @llvm.ctlz.i64(i64 %a, i1 true) readnone
  ret i64 %val
}

; Here we truncate the 64-bit value of LLVM's ctlz intrinsic to 32 bits, the
; natural return width of ptx's clz.b64 instruction.  No conversions should be
; necessary in the PTX.
; CHECK-LABEL: myctlz64_as_32(
define i32 @myctlz64_as_32(i64 %a) {
; CHECK: ld.param.
; CHECK-NEXT: clz.b64
; CHECK-NEXT: st.param.
; CHECK-NEXT: ret;
  %val = call i64 @llvm.ctlz.i64(i64 %a, i1 false) readnone
  %trunc = trunc i64 %val to i32
  ret i32 %trunc
}
; CHECK-LABEL: myctlz64_as_32_2(
define i32 @myctlz64_as_32_2(i64 %a) {
; CHECK: ld.param.
; CHECK-NEXT: clz.b64
; CHECK-NEXT: st.param.
; CHECK-NEXT: ret;
  %val = call i64 @llvm.ctlz.i64(i64 %a, i1 false) readnone
  %trunc = trunc i64 %val to i32
  ret i32 %trunc
}

; ctlz.i16 is implemented by extending the input to i32, computing the result,
; and then truncating the result back down to i16.  But the NVPTX ABI
; zero-extends i16 return values to i32, so the final truncation doesn't appear
; in this function.
; CHECK-LABEL: myctlz_ret16(
define i16 @myctlz_ret16(i16 %a) {
; CHECK: ld.param.
; CHECK-NEXT: cvt.u32.u16
; CHECK-NEXT: clz.b32
; CHECK-NEXT: sub.
; CHECK-NEXT: st.param.
; CHECK-NEXT: ret;
  %val = call i16 @llvm.ctlz.i16(i16 %a, i1 false) readnone
  ret i16 %val
}
; CHECK-LABEL: myctlz_ret16_2(
define i16 @myctlz_ret16_2(i16 %a) {
; CHECK: ld.param.
; CHECK-NEXT: cvt.u32.u16
; CHECK-NEXT: clz.b32
; CHECK-NEXT: sub.
; CHECK-NEXT: st.param.
; CHECK-NEXT: ret;
  %val = call i16 @llvm.ctlz.i16(i16 %a, i1 true) readnone
  ret i16 %val
}

; Here we store the result of ctlz.16 into an i16 pointer, so the trunc should
; remain.
; CHECK-LABEL: myctlz_store16(
define void @myctlz_store16(i16 %a, ptr %b) {
; CHECK: ld.param.
; CHECK-NEXT: cvt.u32.u16
; CHECK-NEXT: clz.b32
; CHECK-DAG: cvt.u16.u32
; CHECK-DAG: sub.
; CHECK: st.{{[a-z]}}16
; CHECK: ret;
  %val = call i16 @llvm.ctlz.i16(i16 %a, i1 false) readnone
  store i16 %val, ptr %b
  ret void
}
; CHECK-LABEL: myctlz_store16_2(
define void @myctlz_store16_2(i16 %a, ptr %b) {
; CHECK: ld.param.
; CHECK-NEXT: cvt.u32.u16
; CHECK-NEXT: clz.b32
; CHECK-DAG: cvt.u16.u32
; CHECK-DAG: sub.
; CHECK: st.{{[a-z]}}16
; CHECK: ret;
  %val = call i16 @llvm.ctlz.i16(i16 %a, i1 false) readnone
  store i16 %val, ptr %b
  ret void
}
