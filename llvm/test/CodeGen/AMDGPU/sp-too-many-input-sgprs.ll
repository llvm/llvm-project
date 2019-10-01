; RUN: llc -mtriple=amdgcn-mesa-mesa3d -verify-machineinstrs < %s | FileCheck -check-prefixes=MESA3D,ALL %s
; RUN: llc -mtriple=amdgcn-- -verify-machineinstrs < %s | FileCheck -check-prefixes=UNKNOWN,ALL %s

; Make sure shaders pick a workable SP with > 32 input SGPRs.
; FIXME: Doesn't seem to be getting initial value from right register?

; ALL-LABEL: {{^}}too_many_input_sgprs_32:
; MESA3D-NOT: s34
; MESA3D: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s34 offset:4

; Happens to end up in s32 anyway
; UNKNOWN-NOT: s32
; UNKNOWN: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s32 offset:4
define amdgpu_ps i32 @too_many_input_sgprs_32(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, i32 inreg %arg3, i32 inreg %arg4, i32 inreg %arg5, i32 inreg %arg6, i32 inreg %arg7,
                                              i32 inreg %arg8, i32 inreg %arg9, i32 inreg %arg10, i32 inreg %arg11, i32 inreg %arg12, i32 inreg %arg13, i32 inreg %arg14, i32 inreg %arg15,
                                              i32 inreg %arg16, i32 inreg %arg17, i32 inreg %arg18, i32 inreg %arg19, i32 inreg %arg20, i32 inreg %arg21, i32 inreg %arg22, i32 inreg %arg23,
                                              i32 inreg %arg24, i32 inreg %arg25, i32 inreg %arg26, i32 inreg %arg27, i32 inreg %arg28, i32 inreg %arg29, i32 inreg %arg30, i32 inreg %arg31) {
bb:
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  %tmp = add i32 %arg, %arg1
  %tmp32 = add i32 %tmp, %arg2
  %tmp33 = add i32 %tmp32, %arg3
  %tmp34 = add i32 %tmp33, %arg4
  %tmp35 = add i32 %tmp34, %arg5
  %tmp36 = add i32 %tmp35, %arg6
  %tmp37 = add i32 %tmp36, %arg7
  %tmp38 = add i32 %tmp37, %arg8
  %tmp39 = add i32 %tmp38, %arg9
  %tmp40 = add i32 %tmp39, %arg10
  %tmp41 = add i32 %tmp40, %arg11
  %tmp42 = add i32 %tmp41, %arg12
  %tmp43 = add i32 %tmp42, %arg13
  %tmp44 = add i32 %tmp43, %arg14
  %tmp45 = add i32 %tmp44, %arg15
  %tmp46 = add i32 %tmp45, %arg16
  %tmp47 = add i32 %tmp46, %arg17
  %tmp48 = add i32 %tmp47, %arg18
  %tmp49 = add i32 %tmp48, %arg19
  %tmp50 = add i32 %tmp49, %arg20
  %tmp51 = add i32 %tmp50, %arg21
  %tmp52 = add i32 %tmp51, %arg22
  %tmp53 = add i32 %tmp52, %arg23
  %tmp54 = add i32 %tmp53, %arg24
  %tmp55 = add i32 %tmp54, %arg25
  %tmp56 = add i32 %tmp55, %arg26
  %tmp57 = add i32 %tmp56, %arg27
  %tmp58 = add i32 %tmp57, %arg28
  %tmp59 = add i32 %tmp58, %arg29
  %tmp60 = add i32 %tmp59, %arg30
  %tmp61 = add i32 %tmp60, %arg31
  ret i32 %tmp61
}

; ALL-LABEL: {{^}}too_many_input_sgprs_33:
; MESA3D-NOT: s35
; MESA3D: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s35 offset:4

; UNKNOWN-NOT: s33
; UNKNOWN: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s33 offset:4
define amdgpu_ps i32 @too_many_input_sgprs_33(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, i32 inreg %arg3, i32 inreg %arg4, i32 inreg %arg5, i32 inreg %arg6, i32 inreg %arg7,
                                              i32 inreg %arg8, i32 inreg %arg9, i32 inreg %arg10, i32 inreg %arg11, i32 inreg %arg12, i32 inreg %arg13, i32 inreg %arg14, i32 inreg %arg15,
                                              i32 inreg %arg16, i32 inreg %arg17, i32 inreg %arg18, i32 inreg %arg19, i32 inreg %arg20, i32 inreg %arg21, i32 inreg %arg22, i32 inreg %arg23,
                                              i32 inreg %arg24, i32 inreg %arg25, i32 inreg %arg26, i32 inreg %arg27, i32 inreg %arg28, i32 inreg %arg29, i32 inreg %arg30, i32 inreg %arg31,
                                              i32 inreg %arg32) {
bb:
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  %tmp = add i32 %arg, %arg1
  %tmp32 = add i32 %tmp, %arg2
  %tmp33 = add i32 %tmp32, %arg3
  %tmp34 = add i32 %tmp33, %arg4
  %tmp35 = add i32 %tmp34, %arg5
  %tmp36 = add i32 %tmp35, %arg6
  %tmp37 = add i32 %tmp36, %arg7
  %tmp38 = add i32 %tmp37, %arg8
  %tmp39 = add i32 %tmp38, %arg9
  %tmp40 = add i32 %tmp39, %arg10
  %tmp41 = add i32 %tmp40, %arg11
  %tmp42 = add i32 %tmp41, %arg12
  %tmp43 = add i32 %tmp42, %arg13
  %tmp44 = add i32 %tmp43, %arg14
  %tmp45 = add i32 %tmp44, %arg15
  %tmp46 = add i32 %tmp45, %arg16
  %tmp47 = add i32 %tmp46, %arg17
  %tmp48 = add i32 %tmp47, %arg18
  %tmp49 = add i32 %tmp48, %arg19
  %tmp50 = add i32 %tmp49, %arg20
  %tmp51 = add i32 %tmp50, %arg21
  %tmp52 = add i32 %tmp51, %arg22
  %tmp53 = add i32 %tmp52, %arg23
  %tmp54 = add i32 %tmp53, %arg24
  %tmp55 = add i32 %tmp54, %arg25
  %tmp56 = add i32 %tmp55, %arg26
  %tmp57 = add i32 %tmp56, %arg27
  %tmp58 = add i32 %tmp57, %arg28
  %tmp59 = add i32 %tmp58, %arg29
  %tmp60 = add i32 %tmp59, %arg30
  %tmp61 = add i32 %tmp60, %arg31
  %tmp62 = add i32 %tmp61, %arg32
  ret i32 %tmp62
}
