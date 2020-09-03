; RUN: not llc -march=r600 < %s 2>&1 | FileCheck -check-prefix=ERROR %s
; RUN: llc -march=amdgcn < %s | FileCheck -check-prefix=GCN %s

declare hidden i32 @memcmp(i8 addrspace(1)* readonly nocapture, i8 addrspace(1)* readonly nocapture, i64) #0
declare hidden i8 addrspace(1)* @memchr(i8 addrspace(1)* readonly nocapture, i32, i64) #1
declare hidden i8* @strcpy(i8* nocapture, i8* readonly nocapture) #0
declare hidden i32 @strlen(i8* nocapture) #1
declare hidden i32 @strnlen(i8* nocapture, i32) #1
declare hidden i32 @strcmp(i8* nocapture, i8* nocapture) #1


; ERROR: error: <unknown>:0:0: in function test_memcmp void (i8 addrspace(1)*, i8 addrspace(1)*, i32*): unsupported call to function memcmp

; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, memcmp@rel32@lo+4
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, memcmp@rel32@hi+12
define amdgpu_kernel void @test_memcmp(i8 addrspace(1)* %x, i8 addrspace(1)* %y, i32* nocapture %p) #0 {
entry:
  %cmp = tail call i32 @memcmp(i8 addrspace(1)* %x, i8 addrspace(1)* %y, i64 2)
  store volatile i32 %cmp, i32 addrspace(1)* undef
  ret void
}

; ERROR: error: <unknown>:0:0: in function test_memchr void (i8 addrspace(1)*, i32, i64): unsupported call to function memchr

; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, memchr@rel32@lo+4
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, memchr@rel32@hi+12
define amdgpu_kernel void @test_memchr(i8 addrspace(1)* %src, i32 %char, i64 %len) #0 {
  %res = call i8 addrspace(1)* @memchr(i8 addrspace(1)* %src, i32 %char, i64 %len)
  store volatile i8 addrspace(1)* %res, i8 addrspace(1)* addrspace(1)* undef
  ret void
}

; ERROR: error: <unknown>:0:0: in function test_strcpy void (i8*, i8*): unsupported call to function strcpy

; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, strcpy@rel32@lo+4
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, strcpy@rel32@hi+12
define amdgpu_kernel void @test_strcpy(i8* %dst, i8* %src) #0 {
  %res = call i8* @strcpy(i8* %dst, i8* %src)
  store volatile i8* %res, i8* addrspace(1)* undef
  ret void
}

; ERROR: error: <unknown>:0:0: in function test_strcmp void (i8*, i8*): unsupported call to function strcmp

; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, strcmp@rel32@lo+4
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, strcmp@rel32@hi+12
define amdgpu_kernel void @test_strcmp(i8* %src0, i8* %src1) #0 {
  %res = call i32 @strcmp(i8* %src0, i8* %src1)
  store volatile i32 %res, i32 addrspace(1)* undef
  ret void
}

; ERROR: error: <unknown>:0:0: in function test_strlen void (i8*): unsupported call to function strlen

; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, strlen@rel32@lo+4
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, strlen@rel32@hi+12
define amdgpu_kernel void @test_strlen(i8* %src) #0 {
  %res = call i32 @strlen(i8* %src)
  store volatile i32 %res, i32 addrspace(1)* undef
  ret void
}

; ERROR: error: <unknown>:0:0: in function test_strnlen void (i8*, i32): unsupported call to function strnlen

; GCN: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, strnlen@rel32@lo+4
; GCN: s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, strnlen@rel32@hi+12
define amdgpu_kernel void @test_strnlen(i8* %src, i32 %size) #0 {
  %res = call i32 @strnlen(i8* %src, i32 %size)
  store volatile i32 %res, i32 addrspace(1)* undef
  ret void
}

attributes #0 = { nounwind }
