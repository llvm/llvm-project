; RUN: not llc < %s -march=amdgcn -mcpu=hawaii
; RUN: not llc < %s -march=amdgcn -mcpu=tonga
; RUN: not llc < %s -march=amdgcn -mcpu=gfx900
; RUN: not llc < %s -march=amdgcn -mcpu=gfx1010

; TODO: Add GlobalISel tests, currently it silently miscompiles as GISel does not handle BF16 at all.

; We only have storage-only BF16 support so check codegen fails if we attempt to do operations on bfloats.

define void @test_fneg(bfloat %a, ptr addrspace(1) %out) {
  %result = fneg bfloat %a
  store bfloat %result, ptr addrspace(1) %out
  ret void
}

define void @test_fabs(bfloat %a, ptr addrspace(1) %out) {
  %result = fabs bfloat %a
  store bfloat %result, ptr addrspace(1) %out
  ret void
}

define void @test_add(bfloat %a, bfloat %b, ptr addrspace(1) %out) {
  %result = fadd bfloat %a, %b
  store bfloat %result, ptr addrspace(1) %out
  ret void
}

define void @test_mul(bfloat %a, bfloat %b, ptr addrspace(1) %out) {
  %result = fmul bfloat %a, %b
  store bfloat %result, ptr addrspace(1) %out
  ret void
}
