; XFAIL: *
; RUN: llc -global-isel -mtriple=amdgcn -mcpu=fiji -stop-after=irtranslator -verify-machineinstrs -o - %s

define void @void_func_v2i65(<2 x i65> %arg0) #0 {
  store <2 x i65> %arg0, ptr addrspace(1) undef
  ret void
}
