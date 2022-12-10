; XFAIL: *
; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -mcpu=fiji -stop-after=irtranslator -verify-machineinstrs -o - %s

define <2 x i65> @v2i65_func_void() #0 {
  %val = load <2 x i65>, ptr addrspace(1) undef
  ret <2 x i65> %val
}
