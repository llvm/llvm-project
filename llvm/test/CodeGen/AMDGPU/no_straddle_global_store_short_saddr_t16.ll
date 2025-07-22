;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn -mcpu=gfx1100 -mattr=+real-true16,dumpcode --filetype=obj < %s | llvm-objdump --triple=amdgcn --mcpu=gfx1100 -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis

declare half @llvm.canonicalize.f16(half) #0

define amdgpu_kernel void @test_fold_canonicalize_undef_value_f16(ptr addrspace(1) %out, half %value) #1 {
  %canonA = call half @llvm.canonicalize.f16(half %value)
  %canonB = call half @llvm.canonicalize.f16(half undef)  
  store half %canonA, ptr addrspace(1) %out
  %out2 =  getelementptr half, ptr addrspace(1) %out, i64 10
  store half %canonB, ptr addrspace(1) %out2
  %out3 =  getelementptr half, ptr addrspace(1) %out, i64 3333158
  store half %canonB, ptr addrspace(1) %out3
  ret void
}


attributes #1 = { nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" }

