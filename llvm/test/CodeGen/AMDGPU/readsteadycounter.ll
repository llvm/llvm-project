; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx700 < %s | FileCheck %s -check-prefixes=GCN,GFX700
; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck %s -check-prefixes=GCN,GFX900
; RUN: llc -global-isel=1 -new-reg-bank-select -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck %s -check-prefixes=GCN,GFX900
; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1100 < %s | FileCheck %s -check-prefixes=GCN,GFX1100
; RUN: llc -global-isel=1 -new-reg-bank-select -mtriple=amdgcn -mcpu=gfx1100 < %s | FileCheck %s -check-prefixes=GCN,GFX1100

declare i64 @llvm.readsteadycounter() #0

; GCN-LABEL: {{^}}test_readsteadycounter:
; GFX700: s_mov_b32 s[[REG:[0-9]+]], 0
; GFX900: s_memrealtime s[[[LO:[0-9]+]]:[[HI:[0-9]+]]]
; GFX900: s_memrealtime s[[[LO:[0-9]+]]:[[HI:[0-9]+]]]
; GFX1100: s_sendmsg_rtn_b64 s[[[LO:[0-9]+]]:[[HI:[0-9]+]]], sendmsg(MSG_RTN_GET_REALTIME)
; GFX1100: s_sendmsg_rtn_b64 s[[[LO:[0-9]+]]:[[HI:[0-9]+]]], sendmsg(MSG_RTN_GET_REALTIME)
define amdgpu_kernel void @test_readsteadycounter(ptr addrspace(1) %out) #0 {
  %cycle0 = call i64 @llvm.readsteadycounter()
  store volatile i64 %cycle0, ptr addrspace(1) %out

  %cycle1 = call i64 @llvm.readsteadycounter()
  store volatile i64 %cycle1, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
