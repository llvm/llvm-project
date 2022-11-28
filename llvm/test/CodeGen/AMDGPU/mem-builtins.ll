; RUN: not llc -march=r600 < %s 2>&1 | FileCheck -check-prefix=ERROR %s
; RUN: llc -march=amdgcn < %s | FileCheck -check-prefix=GCN %s

declare hidden i32 @memcmp(ptr addrspace(1) readonly nocapture, ptr addrspace(1) readonly nocapture, i64) #0
declare hidden ptr addrspace(1) @memchr(ptr addrspace(1) readonly nocapture, i32, i64) #1
declare hidden ptr @strcpy(ptr nocapture, ptr readonly nocapture) #0
declare hidden i32 @strlen(ptr nocapture) #1
declare hidden i32 @strnlen(ptr nocapture, i32) #1
declare hidden i32 @strcmp(ptr nocapture, ptr nocapture) #1


; ERROR: error: <unknown>:0:0: in function test_memcmp void (ptr addrspace(1), ptr addrspace(1), ptr): unsupported call to function memcmp

; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, memcmp@rel32@lo+4
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, memcmp@rel32@hi+12
define amdgpu_kernel void @test_memcmp(ptr addrspace(1) %x, ptr addrspace(1) %y, ptr nocapture %p) #0 {
entry:
  %cmp = tail call i32 @memcmp(ptr addrspace(1) %x, ptr addrspace(1) %y, i64 2)
  store volatile i32 %cmp, ptr addrspace(1) undef
  ret void
}

; ERROR: error: <unknown>:0:0: in function test_memchr void (ptr addrspace(1), i32, i64): unsupported call to function memchr

; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, memchr@rel32@lo+4
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, memchr@rel32@hi+12
define amdgpu_kernel void @test_memchr(ptr addrspace(1) %src, i32 %char, i64 %len) #0 {
  %res = call ptr addrspace(1) @memchr(ptr addrspace(1) %src, i32 %char, i64 %len)
  store volatile ptr addrspace(1) %res, ptr addrspace(1) undef
  ret void
}

; ERROR: error: <unknown>:0:0: in function test_strcpy void (ptr, ptr): unsupported call to function strcpy

; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, strcpy@rel32@lo+4
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, strcpy@rel32@hi+12
define amdgpu_kernel void @test_strcpy(ptr %dst, ptr %src) #0 {
  %res = call ptr @strcpy(ptr %dst, ptr %src)
  store volatile ptr %res, ptr addrspace(1) undef
  ret void
}

; ERROR: error: <unknown>:0:0: in function test_strcmp void (ptr, ptr): unsupported call to function strcmp

; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, strcmp@rel32@lo+4
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, strcmp@rel32@hi+12
define amdgpu_kernel void @test_strcmp(ptr %src0, ptr %src1) #0 {
  %res = call i32 @strcmp(ptr %src0, ptr %src1)
  store volatile i32 %res, ptr addrspace(1) undef
  ret void
}

; ERROR: error: <unknown>:0:0: in function test_strlen void (ptr): unsupported call to function strlen

; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, strlen@rel32@lo+4
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, strlen@rel32@hi+12
define amdgpu_kernel void @test_strlen(ptr %src) #0 {
  %res = call i32 @strlen(ptr %src)
  store volatile i32 %res, ptr addrspace(1) undef
  ret void
}

; ERROR: error: <unknown>:0:0: in function test_strnlen void (ptr, i32): unsupported call to function strnlen

; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, strnlen@rel32@lo+4
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, strnlen@rel32@hi+12
define amdgpu_kernel void @test_strnlen(ptr %src, i32 %size) #0 {
  %res = call i32 @strnlen(ptr %src, i32 %size)
  store volatile i32 %res, ptr addrspace(1) undef
  ret void
}

attributes #0 = { nounwind }
