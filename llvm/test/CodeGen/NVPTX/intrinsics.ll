; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; CHECK-LABEL: test_fabsf(
define float @test_fabsf(float %f) {
; CHECK: abs.f32
  %x = call float @llvm.fabs.f32(float %f)
  ret float %x
}

; CHECK-LABEL: test_fabs(
define double @test_fabs(double %d) {
; CHECK: abs.f64
  %x = call double @llvm.fabs.f64(double %d)
  ret double %x
}

; CHECK-LABEL: test_nvvm_sqrt(
define float @test_nvvm_sqrt(float %a) {
; CHECK: sqrt.rn.f32
  %val = call float @llvm.nvvm.sqrt.f(float %a)
  ret float %val
}

; CHECK-LABEL: test_llvm_sqrt(
define float @test_llvm_sqrt(float %a) {
; CHECK: sqrt.rn.f32
  %val = call float @llvm.sqrt.f32(float %a)
  ret float %val
}

; CHECK-LABEL: test_bitreverse32(
define i32 @test_bitreverse32(i32 %a) {
; CHECK: brev.b32
  %val = call i32 @llvm.bitreverse.i32(i32 %a)
  ret i32 %val
}

; CHECK-LABEL: test_bitreverse64(
define i64 @test_bitreverse64(i64 %a) {
; CHECK: brev.b64
  %val = call i64 @llvm.bitreverse.i64(i64 %a)
  ret i64 %val
}

; CHECK-LABEL: test_popc32(
define i32 @test_popc32(i32 %a) {
; CHECK: popc.b32
  %val = call i32 @llvm.ctpop.i32(i32 %a)
  ret i32 %val
}

; CHECK-LABEL: test_popc64
define i64 @test_popc64(i64 %a) {
; CHECK: popc.b64
; CHECK: cvt.u64.u32
  %val = call i64 @llvm.ctpop.i64(i64 %a)
  ret i64 %val
}

; NVPTX popc.b64 returns an i32 even though @llvm.ctpop.i64 returns an i64, so
; if this function returns an i32, there's no need to do any type conversions
; in the ptx.
; CHECK-LABEL: test_popc64_trunc
define i32 @test_popc64_trunc(i64 %a) {
; CHECK: popc.b64
; CHECK-NOT: cvt.
  %val = call i64 @llvm.ctpop.i64(i64 %a)
  %trunc = trunc i64 %val to i32
  ret i32 %trunc
}

; llvm.ctpop.i16 is implemenented by converting to i32, running popc.b32, and
; then converting back to i16.
; CHECK-LABEL: test_popc16
define void @test_popc16(i16 %a, ptr %b) {
; CHECK: cvt.u32.u16
; CHECK: popc.b32
; CHECK: cvt.u16.u32
  %val = call i16 @llvm.ctpop.i16(i16 %a)
  store i16 %val, ptr %b
  ret void
}

; If we call llvm.ctpop.i16 and then zext the result to i32, we shouldn't need
; to do any conversions after calling popc.b32, because that returns an i32.
; CHECK-LABEL: test_popc16_to_32
define i32 @test_popc16_to_32(i16 %a) {
; CHECK: cvt.u32.u16
; CHECK: popc.b32
; CHECK-NOT: cvt.
  %val = call i16 @llvm.ctpop.i16(i16 %a)
  %zext = zext i16 %val to i32
  ret i32 %zext
}

; Most of nvvm.read.ptx.sreg.* intrinsics always return the same value and may
; be CSE'd.
; CHECK-LABEL: test_tid
define i32 @test_tid() {
; CHECK: mov.u32         %r{{.*}}, %tid.x;
  %a = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; CHECK-NOT: mov.u32         %r{{.*}}, %tid.x;
  %b = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ret = add i32 %a, %b
; CHECK: ret
  ret i32 %ret
}

; reading clock() or clock64() should not be CSE'd as each read may return
; different value.
; CHECK-LABEL: test_clock
define i32 @test_clock() {
; CHECK: mov.u32         %r{{.*}}, %clock;
  %a = tail call i32 @llvm.nvvm.read.ptx.sreg.clock()
; CHECK: mov.u32         %r{{.*}}, %clock;
  %b = tail call i32 @llvm.nvvm.read.ptx.sreg.clock()
  %ret = add i32 %a, %b
; CHECK: ret
  ret i32 %ret
}

; CHECK-LABEL: test_clock64
define i64 @test_clock64() {
; CHECK: mov.u64         %r{{.*}}, %clock64;
  %a = tail call i64 @llvm.nvvm.read.ptx.sreg.clock64()
; CHECK: mov.u64         %r{{.*}}, %clock64;
  %b = tail call i64 @llvm.nvvm.read.ptx.sreg.clock64()
  %ret = add i64 %a, %b
; CHECK: ret
  ret i64 %ret
}

%struct.S = type { [4 x i64] }

; CHECK-LABEL: test_memcpy
define dso_local void @test_memcpy(ptr noundef %dst, ptr noundef %src) #0 {
; CHECK-DAG:        ld.param.u{{32|64}}    %[[D:(r|rd)[0-9]+]], [test_memcpy_param_0];
; CHECK-DAG:        ld.param.u{{32|64}}    %[[S:(r|rd)[0-9]+]], [test_memcpy_param_1];
; CHECK-DAG:        ld.u8   %[[V30:rs[0-9]+]], [%[[S]]+30];
; CHECK-DAG:        st.u8   [%[[D]]+30], %[[V30]];
; CHECK-DAG:        ld.u16  %[[V28:rs[0-9]+]], [%[[S]]+28];
; CHECK-DAG:        st.u16  [%[[D]]+28], %[[V28]];
; CHECK-DAG:        ld.u32  %[[V24:r[0-9]+]], [%[[S]]+24];
; CHECK-DAG:        st.u32  [%[[D]]+24], %[[V24]];
; CHECK-DAG:        ld.u64  %[[V16:rd[0-9]+]], [%[[S]]+16];
; CHECK-DAG:        st.u64  [%[[D]]+16], %[[V16]];
; CHECK-DAG:        ld.v4.u32       {[[V0:%r[0-9]+, %r[0-9]+, %r[0-9]+, %r[0-9]+]]}, [%[[S]]];
; CHECK-DAG:        st.v4.u32       [%[[D]]], {[[V0]]};
  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %dst, ptr align 16 %src, i64 31, i1 false)
  ret void
}

; CHECK-LABEL: test_memcpy_a8
define dso_local void @test_memcpy_a8(ptr noundef %dst, ptr noundef %src) #0 {
; CHECK-DAG:        ld.param.u{{32|64}}    %[[D:(r|rd)[0-9]+]], [test_memcpy_a8_param_0];
; CHECK-DAG:        ld.param.u{{32|64}}    %[[S:(r|rd)[0-9]+]], [test_memcpy_a8_param_1];
; CHECK-DAG:        ld.u8   %[[V30:rs[0-9]+]], [%[[S]]+30];
; CHECK-DAG:        st.u8   [%[[D]]+30], %[[V30]];
; CHECK-DAG:        ld.u16  %[[V28:rs[0-9]+]], [%[[S]]+28];
; CHECK-DAG:        st.u16  [%[[D]]+28], %[[V28]];
; CHECK-DAG:        ld.u32  %[[V24:r[0-9]+]], [%[[S]]+24];
; CHECK-DAG:        st.u32  [%[[D]]+24], %[[V24]];
; CHECK-DAG:        ld.u64  %[[V16:rd[0-9]+]], [%[[S]]+16];
; CHECK-DAG:        st.u64  [%[[D]]+16], %[[V16]];
; CHECK-DAG:        ld.u64  %[[V8:rd[0-9]+]], [%[[S]]+8];
; CHECK-DAG:        st.u64  [%[[D]]+8], %[[V8]];
; CHECK-DAG:        ld.u64  %[[V0:rd[0-9]+]], [%[[S]]];
; CHECK-DAG:        st.u64  [%[[D]]], %[[V0]];
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 %src, i64 31, i1 false)
  ret void
}

declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
declare float @llvm.nvvm.sqrt.f(float)
declare float @llvm.sqrt.f32(float)
declare i32 @llvm.bitreverse.i32(i32)
declare i64 @llvm.bitreverse.i64(i64)
declare i16 @llvm.ctpop.i16(i16)
declare i32 @llvm.ctpop.i32(i32)
declare i64 @llvm.ctpop.i64(i64)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.clock()
declare i64 @llvm.nvvm.read.ptx.sreg.clock64()
