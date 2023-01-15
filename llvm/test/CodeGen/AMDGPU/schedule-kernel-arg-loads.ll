; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=FUNC -check-prefix=SI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=FUNC -check-prefix=VI -check-prefix=GCN %s

; FUNC-LABEL: {{^}}cluster_arg_loads:
; SI: s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x9
; SI-NEXT: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0xd

; VI: s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x24
; VI-NEXT: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x34
define amdgpu_kernel void @cluster_arg_loads(ptr addrspace(1) %out0, ptr addrspace(1) %out1, i32 %x, i32 %y) nounwind {
  store i32 %x, ptr addrspace(1) %out0, align 4
  store i32 %y, ptr addrspace(1) %out1, align 4
  ret void
}

; Test for a crash in SIInstrInfo::areLoadsFromSameBasePtr() when
; s_load_dwordx2 has a register offset

; FUNC-LABEL: @same_base_ptr_crash
; GCN: s_load_dwordx2
; GCN: s_load_dwordx2
; GCN: s_load_dwordx2
; GCN: s_endpgm
define amdgpu_kernel void @same_base_ptr_crash(ptr addrspace(1) %out,
    i64 %arg0, i64 %arg1, i64 %arg2, i64 %arg3, i64 %arg4, i64 %arg5, i64 %arg6, i64 %arg7,
    i64 %arg8, i64 %arg9, i64 %arg10, i64 %arg11, i64 %arg12, i64 %arg13, i64 %arg14, i64 %arg15,
    i64 %arg16, i64 %arg17, i64 %arg18, i64 %arg19, i64 %arg20, i64 %arg21, i64 %arg22, i64 %arg23,
    i64 %arg24, i64 %arg25, i64 %arg26, i64 %arg27, i64 %arg28, i64 %arg29, i64 %arg30, i64 %arg31,
    i64 %arg32, i64 %arg33, i64 %arg34, i64 %arg35, i64 %arg36, i64 %arg37, i64 %arg38, i64 %arg39,
    i64 %arg40, i64 %arg41, i64 %arg42, i64 %arg43, i64 %arg44, i64 %arg45, i64 %arg46, i64 %arg47,
    i64 %arg48, i64 %arg49, i64 %arg50, i64 %arg51, i64 %arg52, i64 %arg53, i64 %arg54, i64 %arg55,
    i64 %arg56, i64 %arg57, i64 %arg58, i64 %arg59, i64 %arg60, i64 %arg61, i64 %arg62, i64 %arg63,
    i64 %arg64, i64 %arg65, i64 %arg66, i64 %arg67, i64 %arg68, i64 %arg69, i64 %arg70, i64 %arg71,
    i64 %arg72, i64 %arg73, i64 %arg74, i64 %arg75, i64 %arg76, i64 %arg77, i64 %arg78, i64 %arg79,
    i64 %arg80, i64 %arg81, i64 %arg82, i64 %arg83, i64 %arg84, i64 %arg85, i64 %arg86, i64 %arg87,
    i64 %arg88, i64 %arg89, i64 %arg90, i64 %arg91, i64 %arg92, i64 %arg93, i64 %arg94, i64 %arg95,
    i64 %arg96, i64 %arg97, i64 %arg98, i64 %arg99, i64 %arg100, i64 %arg101, i64 %arg102, i64 %arg103,
    i64 %arg104, i64 %arg105, i64 %arg106, i64 %arg107, i64 %arg108, i64 %arg109, i64 %arg110, i64 %arg111,
    i64 %arg112, i64 %arg113, i64 %arg114, i64 %arg115, i64 %arg116, i64 %arg117, i64 %arg118, i64 %arg119,
    i64 %arg120, i64 %arg121, i64 %arg122, i64 %arg123, i64 %arg124, i64 %arg125, i64 %arg126) {
entry:
  %value = add i64 %arg124, %arg126
  store i64 %value, ptr addrspace(1) %out, align 8
  ret void
}
