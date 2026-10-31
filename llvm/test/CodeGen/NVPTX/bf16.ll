; RUN: llc < %s -mtriple=nvptx64 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 | %ptxas-verify %}

; LDST: .b8 bfloat_array[8] = {1, 2, 3, 4, 5, 6, 7, 8};
@"bfloat_array" = addrspace(1) constant [4 x bfloat]
                [bfloat 0xR0201, bfloat 0xR0403, bfloat 0xR0605, bfloat 0xR0807]

define void @test_load_store(ptr addrspace(1) %in, ptr addrspace(1) %out) {
; CHECK-LABEL: @test_load_store
; CHECK: ld.global.b16 [[TMP:%rs[0-9]+]], [{{%rd[0-9]+}}]
; CHECK: st.global.b16 [{{%rd[0-9]+}}], [[TMP]]
  %val = load bfloat, ptr addrspace(1) %in
  store bfloat %val, ptr addrspace(1) %out
  ret void
}

define void @test_bitcast_from_bfloat(ptr addrspace(1) %in, ptr addrspace(1) %out) {
; CHECK-LABEL: @test_bitcast_from_bfloat
; CHECK: ld.global.b16 [[TMP:%rs[0-9]+]], [{{%rd[0-9]+}}]
; CHECK: st.global.b16 [{{%rd[0-9]+}}], [[TMP]]
  %val = load bfloat, ptr addrspace(1) %in
  %val_int = bitcast bfloat %val to i16
  store i16 %val_int, ptr addrspace(1) %out
  ret void
}

define void @test_bitcast_to_bfloat(ptr addrspace(1) %out, ptr addrspace(1) %in) {
; CHECK-LABEL: @test_bitcast_to_bfloat
; CHECK: ld.global.b16 [[TMP:%rs[0-9]+]], [{{%rd[0-9]+}}]
; CHECK: st.global.b16 [{{%rd[0-9]+}}], [[TMP]]
  %val = load i16, ptr addrspace(1) %in
  %val_fp = bitcast i16 %val to bfloat
  store bfloat %val_fp, ptr addrspace(1) %out
  ret void
}

; Combining the byte concatenation can produce a v2bf16 SCALAR_TO_VECTOR.
define <4 x i8> @scalar_to_vector_bf16(bfloat %x) {
; CHECK-LABEL: scalar_to_vector_bf16(
; CHECK:       {
; CHECK-NEXT:    .reg .b16 %rs<2>;
; CHECK-NEXT:    .reg .b32 %r<2>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.b16 %rs1, [scalar_to_vector_bf16_param_0];
; CHECK-NEXT:    mov.b32 %r1, {%rs1, 0};
; CHECK-NEXT:    st.param.b32 [func_retval0], %r1;
; CHECK-NEXT:    ret;
  %bytes = bitcast bfloat %x to <2 x i8>
  %wide = shufflevector <2 x i8> %bytes, <2 x i8> poison,
                       <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i8> %wide
}

; The same node can be introduced when expanding an unaligned bfloat load.
define <4 x i8> @scalar_to_vector_bf16_unaligned(ptr addrspace(1) %in) {
; CHECK-LABEL: scalar_to_vector_bf16_unaligned(
; CHECK:       {
; CHECK-NEXT:    .reg .b16 %rs<5>;
; CHECK-NEXT:    .reg .b32 %r<2>;
; CHECK-NEXT:    .reg .b64 %rd<2>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.b64 %rd1, [scalar_to_vector_bf16_unaligned_param_0];
; CHECK-NEXT:    ld.global.b8 %rs1, [%rd1];
; CHECK-NEXT:    ld.global.b8 %rs2, [%rd1+1];
; CHECK-NEXT:    shl.b16 %rs3, %rs2, 8;
; CHECK-NEXT:    or.b16 %rs4, %rs3, %rs1;
; CHECK-NEXT:    mov.b32 %r1, {%rs4, 0};
; CHECK-NEXT:    st.param.b32 [func_retval0], %r1;
; CHECK-NEXT:    ret;
  %x = load bfloat, ptr addrspace(1) %in, align 1
  %bytes = bitcast bfloat %x to <2 x i8>
  %wide = shufflevector <2 x i8> %bytes, <2 x i8> poison,
                       <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i8> %wide
}

; Preserve all four bytes when copying through an aligned local array.
define ptx_kernel void @copy_unaligned_bf16(ptr addrspace(1) %in, ptr addrspace(1) %out) {
; CHECK-LABEL: copy_unaligned_bf16(
; CHECK:       {
; CHECK-NEXT:    .reg .b16 %rs<5>;
; CHECK-NEXT:    .reg .b32 %r<12>;
; CHECK-NEXT:    .reg .b64 %rd<3>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.b64 %rd1, [copy_unaligned_bf16_param_0];
; CHECK-NEXT:    ld.param.b64 %rd2, [copy_unaligned_bf16_param_1];
; CHECK-NEXT:    ld.global.b8 %r1, [%rd1];
; CHECK-NEXT:    prmt.b32 %r2, %r1, %r3, 0x3340U;
; CHECK-NEXT:    prmt.b32 %r4, %r5, %r6, 0x3340U;
; CHECK-NEXT:    prmt.b32 %r7, %r2, %r4, 0x5410U;
; CHECK-NEXT:    ld.global.b8 %rs1, [%rd1+1];
; CHECK-NEXT:    ld.global.b8 %rs2, [%rd1+2];
; CHECK-NEXT:    shl.b16 %rs3, %rs2, 8;
; CHECK-NEXT:    or.b16 %rs4, %rs3, %rs1;
; CHECK-NEXT:    mov.b32 %r8, {%rs4, 0};
; CHECK-NEXT:    prmt.b32 %r9, %r7, %r8, 0x540U;
; CHECK-NEXT:    ld.global.b8 %r10, [%rd1+3];
; CHECK-NEXT:    bfi.b32 %r11, %r10, %r9, 24, 8;
; CHECK-NEXT:    st.global.b32 [%rd2], %r11;
; CHECK-NEXT:    ret;
  %loc = alloca [4 x i8], align 4
  %a = load i8, ptr addrspace(1) %in, align 1
  store i8 %a, ptr %loc, align 1
  %pi = getelementptr inbounds i8, ptr addrspace(1) %in, i64 1
  %b = load bfloat, ptr addrspace(1) %pi, align 1
  %pl = getelementptr inbounds i8, ptr %loc, i64 1
  store bfloat %b, ptr %pl, align 1
  %pi3 = getelementptr inbounds i8, ptr addrspace(1) %in, i64 3
  %c = load i8, ptr addrspace(1) %pi3, align 1
  %pl3 = getelementptr inbounds i8, ptr %loc, i64 3
  store i8 %c, ptr %pl3, align 1
  %v = load <2 x i16>, ptr %loc, align 4
  store <2 x i16> %v, ptr addrspace(1) %out, align 4
  ret void
}
