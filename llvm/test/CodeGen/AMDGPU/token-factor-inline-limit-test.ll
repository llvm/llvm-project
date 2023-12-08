; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-TFILD %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -combiner-tokenfactor-inline-limit=7 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GCN-TFIL7 %s


; GCN-LABEL: {{^}}token_factor_inline_limit_test:

; GCN-TFLID: v_mov_b32_e32 [[REG7:v[0-9]+]], 7
; GCN-TFLID: buffer_store_dword [[REG7]], {{.*$}}
; GCN-TFILD: v_mov_b32_e32 [[REG8:v[0-9]+]], 8
; GCN-TFILD: buffer_store_dword [[REG8]], {{.*}} offset:4
; GCN-TFILD: v_mov_b32_e32 [[REG9:v[0-9]+]], 9
; GCN-TFILD: buffer_store_dword [[REG9]], {{.*}} offset:8
; GCN-TFILD: v_mov_b32_e32 [[REG10:v[0-9]+]], 10
; GCN-TFILD: buffer_store_dword [[REG10]], {{.*}} offset:12
; GCN-TFILD: v_mov_b32_e32 [[REG11:v[0-9]+]], 11
; GCN-TFILD: buffer_store_dword [[REG11]], {{.*}} offset:16
; GCN-TFILD: v_mov_b32_e32 [[REG12:v[0-9]+]], 12
; GCN-TFILD: buffer_store_dword [[REG12]], {{.*}} offset:20
; GCN-TFILD: v_mov_b32_e32 [[REG13:v[0-9]+]], 13
; GCN-TFILD: buffer_store_dword [[REG13]], {{.*}} offset:24
; GCN-TFILD: v_mov_b32_e32 [[REG14:v[0-9]+]], 14
; GCN-TFILD: buffer_store_dword [[REG14]], {{.*}} offset:28
; GCN-TFILD: v_mov_b32_e32 [[REG15:v[0-9]+]], 15
; GCN-TFILD: buffer_store_dword [[REG15]], {{.*}} offset:32

; GCN-TFIL7: v_mov_b32_e32 [[REG15:v[0-9]+]], 15
; GCN-TFIL7: buffer_store_dword [[REG15]], {{.*}} offset:32
; GCN-TFIL7: v_mov_b32_e32 [[REG14:v[0-9]+]], 14
; GCN-TFIL7: buffer_store_dword [[REG14]], {{.*}} offset:28
; GCN-TFIL7: v_mov_b32_e32 [[REG13:v[0-9]+]], 13
; GCN-TFIL7: buffer_store_dword [[REG13]], {{.*}} offset:24
; GCN-TFIL7: v_mov_b32_e32 [[REG12:v[0-9]+]], 12
; GCN-TFIL7: buffer_store_dword [[REG12]], {{.*}} offset:20
; GCN-TFIL7: v_mov_b32_e32 [[REG11:v[0-9]+]], 11
; GCN-TFIL7: buffer_store_dword [[REG11]], {{.*}} offset:16
; GCN-TFIL7: v_mov_b32_e32 [[REG10:v[0-9]+]], 10
; GCN-TFIL7: buffer_store_dword [[REG10]], {{.*}} offset:12
; GCN-TFIL7: v_mov_b32_e32 [[REG9:v[0-9]+]], 9
; GCN-TFIL7: buffer_store_dword [[REG9]], {{.*}} offset:8
; GCN-TFIL7: v_mov_b32_e32 [[REG8:v[0-9]+]], 8
; GCN-TFIL7: buffer_store_dword [[REG8]], {{.*}} offset:4
; GCN-TFLL7: v_mov_b32_e32 [[REG7:v[0-9]+]], 7
; GCN-TFLL7: buffer_store_dword [[REG7]], {{.*$}}

; GCN: s_getpc
define void @token_factor_inline_limit_test() {
entry:
  call void @external_void_func_8xv5i32(
      <5 x i32><i32 0, i32 0, i32 0, i32 0, i32 0>,
      <5 x i32><i32 1, i32 1, i32 1, i32 1, i32 1>,
      <5 x i32><i32 2, i32 2, i32 2, i32 2, i32 2>,
      <5 x i32><i32 3, i32 3, i32 3, i32 3, i32 3>,
      <5 x i32><i32 4, i32 4, i32 4, i32 4, i32 4>,
      <5 x i32><i32 5, i32 5, i32 5, i32 5, i32 5>,
      <5 x i32><i32 6, i32 7, i32 8, i32 9, i32 10>,
      <5 x i32><i32 11, i32 12, i32 13, i32 14, i32 15>)
  ret void
}

declare hidden void @external_void_func_8xv5i32(<5 x i32>, <5 x i32>, <5 x i32>, <5 x i32>,
                                                <5 x i32>, <5 x i32>, <5 x i32>, <5 x i32>)
