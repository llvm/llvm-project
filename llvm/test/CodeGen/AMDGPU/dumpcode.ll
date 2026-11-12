; RUN: llc -mtriple=amdgpu10.10 -mattr=dumpcode -filetype=obj < %s | llvm-objcopy --dump-section .AMDGPU.disasm=- - /dev/null | FileCheck %s -check-prefix=GFX10

; GFX10: f:
; GFX10-NEXT: BB0_0:
; GFX10-NEXT:   s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; BF8C0000
; GFX10-NEXT:   v_mov_b32_e32 v2, 0xde                  ; 7E0402FF 000000DE
; GFX10-NEXT:   global_store_dword v[0:1], v2, off      ; DC708000 007D0200
; GFX10-NEXT:   s_setpc_b64 s[30:31]                    ; BE80201E

define void @f(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %val) {
entry:
  br label %body
body:
  %i = phi i32 [0, %entry], [%inc, %body]
  store i32 222, ptr addrspace(1) %out
  %cmp = icmp ne i32 %i, %val
  %inc = add i32 %i, 1
  br i1 %cmp, label %body, label %end
end:
  ret void
}
