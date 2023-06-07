; RUN: llc < %s -march=nvptx64 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 | %ptxas-verify %}

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
; CHECK: ld.global.u16 [[TMP:%rs[0-9]+]], [{{%rd[0-9]+}}]
; CHECK: st.global.u16 [{{%rd[0-9]+}}], [[TMP]]
  %val = load i16, ptr addrspace(1) %in
  %val_fp = bitcast i16 %val to bfloat
  store bfloat %val_fp, ptr addrspace(1) %out
  ret void
}
