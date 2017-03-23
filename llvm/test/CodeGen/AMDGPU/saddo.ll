; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs< %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs< %s

declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) nounwind readnone
declare { i64, i1 } @llvm.sadd.with.overflow.i64(i64, i64) nounwind readnone

; FUNC-LABEL: {{^}}saddo_i64_zext:
define amdgpu_kernel void @saddo_i64_zext(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %sadd = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %a, i64 %b) nounwind
  %val = extractvalue { i64, i1 } %sadd, 0
  %carry = extractvalue { i64, i1 } %sadd, 1
  %ext = zext i1 %carry to i64
  %add2 = add i64 %val, %ext
  store i64 %add2, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_saddo_i32:
define amdgpu_kernel void @s_saddo_i32(i32 addrspace(1)* %out, i1 addrspace(1)* %carryout, i32 %a, i32 %b) nounwind {
  %sadd = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b) nounwind
  %val = extractvalue { i32, i1 } %sadd, 0
  %carry = extractvalue { i32, i1 } %sadd, 1
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; FUNC-LABEL: {{^}}v_saddo_i32:
define amdgpu_kernel void @v_saddo_i32(i32 addrspace(1)* %out, i1 addrspace(1)* %carryout, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %a = load i32, i32 addrspace(1)* %aptr, align 4
  %b = load i32, i32 addrspace(1)* %bptr, align 4
  %sadd = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b) nounwind
  %val = extractvalue { i32, i1 } %sadd, 0
  %carry = extractvalue { i32, i1 } %sadd, 1
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; FUNC-LABEL: {{^}}s_saddo_i64:
define amdgpu_kernel void @s_saddo_i64(i64 addrspace(1)* %out, i1 addrspace(1)* %carryout, i64 %a, i64 %b) nounwind {
  %sadd = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %a, i64 %b) nounwind
  %val = extractvalue { i64, i1 } %sadd, 0
  %carry = extractvalue { i64, i1 } %sadd, 1
  store i64 %val, i64 addrspace(1)* %out, align 8
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; FUNC-LABEL: {{^}}v_saddo_i64:
; SI: v_add_i32
; SI: v_addc_u32
define amdgpu_kernel void @v_saddo_i64(i64 addrspace(1)* %out, i1 addrspace(1)* %carryout, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr) nounwind {
  %a = load i64, i64 addrspace(1)* %aptr, align 4
  %b = load i64, i64 addrspace(1)* %bptr, align 4
  %sadd = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %a, i64 %b) nounwind
  %val = extractvalue { i64, i1 } %sadd, 0
  %carry = extractvalue { i64, i1 } %sadd, 1
  store i64 %val, i64 addrspace(1)* %out, align 8
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}
