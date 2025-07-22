;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn -mcpu=gfx900  -mattr=dumpcode --filetype=obj < %s | llvm-objdump --triple=amdgcn --mcpu=gfx900 -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis

;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn -mcpu=gfx908  -mattr=dumpcode --filetype=obj < %s | llvm-objdump --triple=amdgcn --mcpu=gfx908 -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis

define amdgpu_ps <4 x float> @test1(<8 x i32> inreg %rsrc, i32 %c) {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 15, i32 %c, <8 x i32> %rsrc, i32 0, i32 0)
  call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %tex, i32 15, i32 %c, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %tex
}

define amdgpu_ps <4 x float> @test2(i32 inreg, i32 inreg, i32 inreg, i32 inreg %m0, <8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, <2 x float> %pos) #6 {
main_body:
  %inst23 = extractelement <2 x float> %pos, i32 0
  %inst24 = extractelement <2 x float> %pos, i32 1
  %inst25 = tail call float @llvm.amdgcn.interp.p1(float %inst23, i32 0, i32 0, i32 %m0)
  %inst26 = tail call float @llvm.amdgcn.interp.p2(float %inst25, float %inst24, i32 0, i32 0, i32 %m0)
  %inst28 = tail call float @llvm.amdgcn.interp.p1(float %inst23, i32 1, i32 0, i32 %m0)
  %inst29 = tail call float @llvm.amdgcn.interp.p2(float %inst28, float %inst24, i32 1, i32 0, i32 %m0)
  %tex = call <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 15, float %inst26, float %inst29, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  ret <4 x float> %tex
}

define amdgpu_ps <4 x float> @test3(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, float %c) {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %tex.1 = bitcast <4 x float> %tex to <4 x i32>
  %tex.2 = extractelement <4 x i32> %tex.1, i32 0

  call void @llvm.amdgcn.struct.buffer.store.v4f32(<4 x float> %tex, <4 x i32> poison, i32 %tex.2, i32 0, i32 0, i32 0)

  ret <4 x float> %tex
}

define amdgpu_ps <4 x float> @test3_ptr_buf(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, float %c) {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %tex.1 = bitcast <4 x float> %tex to <4 x i32>
  %tex.2 = extractelement <4 x i32> %tex.1, i32 0

  call void @llvm.amdgcn.struct.ptr.buffer.store.v4f32(<4 x float> %tex, ptr addrspace(8) poison, i32 %tex.2, i32 0, i32 0, i32 0)

  ret <4 x float> %tex
}

define amdgpu_ps void @test3x(i32 inreg, i32 inreg, i32 inreg, i32 inreg %m0, <8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, <2 x float> %pos) #6 {
main_body:
  %inst23 = extractelement <2 x float> %pos, i32 0
  %inst24 = extractelement <2 x float> %pos, i32 1
  %inst25 = tail call float @llvm.amdgcn.interp.p1(float %inst23, i32 0, i32 0, i32 %m0)
  %inst26 = tail call float @llvm.amdgcn.interp.p2(float %inst25, float %inst24, i32 0, i32 0, i32 %m0)
  %inst28 = tail call float @llvm.amdgcn.interp.p1(float %inst23, i32 1, i32 0, i32 %m0)
  %inst29 = tail call float @llvm.amdgcn.interp.p2(float %inst28, float %inst24, i32 1, i32 0, i32 %m0)
  %tex = call <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 15, float %inst26, float %inst29, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %tex.0 = extractelement <4 x float> %tex, i32 0
  %tex.1 = extractelement <4 x float> %tex, i32 1
  %tex.2 = extractelement <4 x float> %tex, i32 2
  %tex.3 = extractelement <4 x float> %tex, i32 3
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tex.0, float %tex.1, float %tex.2, float %tex.3, i1 true, i1 true)
  ret void
}

define amdgpu_ps <4 x float> @test4(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, ptr addrspace(1) inreg %ptr, i32 %c, i32 %d, float %data) {
main_body:
  %c.1 = mul i32 %c, %d

  call void @llvm.amdgcn.struct.buffer.store.v4f32(<4 x float> poison, <4 x i32> poison, i32 %c.1, i32 0, i32 0, i32 0)
  %c.1.bc = bitcast i32 %c.1 to float
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c.1.bc, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  ret <4 x float> %dtex
}

define amdgpu_ps <4 x float> @test4_ptr_buf(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, ptr addrspace(1) inreg %ptr, i32 %c, i32 %d, float %data) {
main_body:
  %c.1 = mul i32 %c, %d

  call void @llvm.amdgcn.struct.ptr.buffer.store.v4f32(<4 x float> poison, ptr addrspace(8) poison, i32 %c.1, i32 0, i32 0, i32 0)
  %c.1.bc = bitcast i32 %c.1 to float
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c.1.bc, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  ret <4 x float> %dtex
}

define amdgpu_ps float @test5(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> poison, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> poison, i32 %idx1, i32 0, i32 0, i32 0)
  %out = fadd float %src0, %src1
  %out.0 = call float @llvm.amdgcn.wqm.f32(float %out)
  ret float %out.0
}

define amdgpu_ps float @test5_ptr_buf(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx1, i32 0, i32 0, i32 0)
  %out = fadd float %src0, %src1
  %out.0 = call float @llvm.amdgcn.wqm.f32(float %out)
  ret float %out.0
}

define amdgpu_ps float @test6(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> poison, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> poison, i32 %idx1, i32 0, i32 0, i32 0)
  %out = fadd float %src0, %src1
  %out.0 = bitcast float %out to i32
  %out.1 = call i32 @llvm.amdgcn.wqm.i32(i32 %out.0)
  %out.2 = bitcast i32 %out.1 to float
  ret float %out.2
}

define amdgpu_ps float @test6_ptr_buf(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx1, i32 0, i32 0, i32 0)
  %out = fadd float %src0, %src1
  %out.0 = bitcast float %out to i32
  %out.1 = call i32 @llvm.amdgcn.wqm.i32(i32 %out.0)
  %out.2 = bitcast i32 %out.1 to float
  ret float %out.2
}


define amdgpu_ps float @test_wwm1(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx1, i32 0, i32 0, i32 0)
  %out = fadd float %src0, %src1
  %out.0 = call float @llvm.amdgcn.wwm.f32(float %out)
  ret float %out.0
}

define amdgpu_ps float @test_wwm2(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx1, i32 0, i32 0, i32 0)
  %src0.0 = bitcast float %src0 to i32
  %src1.0 = bitcast float %src1 to i32
  %out = add i32 %src0.0, %src1.0
  %out.0 = call i32 @llvm.amdgcn.wwm.i32(i32 %out)
  %out.1 = bitcast i32 %out.0 to float
  ret float %out.1
}

define amdgpu_ps float @test_wwm3(i32 inreg %idx) {
main_body:
  ; use mbcnt to make sure the branch is divergent
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %cc = icmp uge i32 %hi, 16
  br i1 %cc, label %endif, label %if

if:
  %src = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx, i32 0, i32 0, i32 0)
  %out = fadd float %src, %src
  %out.0 = call float @llvm.amdgcn.wwm.f32(float %out)
  %out.1 = fadd float %src, %out.0
  br label %endif

endif:
  %out.2 = phi float [ %out.1, %if ], [ 0.0, %main_body ]
  ret float %out.2
}

define amdgpu_ps float @test_wwm4(i32 inreg %idx) {
main_body:
  ; use mbcnt to make sure the branch is divergent
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %cc = icmp uge i32 %hi, 16
  br i1 %cc, label %endif, label %if

if:
  %src = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx, i32 0, i32 0, i32 0)
  %out = fadd float %src, %src
  %out.0 = call float @llvm.amdgcn.wwm.f32(float %out)
  br label %endif

endif:
  %out.1 = phi float [ %out.0, %if ], [ 0.0, %main_body ]
  ret float %out.1
}

define amdgpu_ps float @test_wwm5(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %src0, ptr addrspace(8) poison, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx1, i32 0, i32 0, i32 0)
  %temp = fadd float %src1, %src1
  %temp.0 = call float @llvm.amdgcn.wwm.f32(float %temp)
  %out = fadd float %temp.0, %temp.0
  %out.0 = call float @llvm.amdgcn.wqm.f32(float %out)
  ret float %out.0
}

define amdgpu_ps float @test_wwm6_then() {
main_body:
  %src0 = load volatile float, ptr addrspace(1) poison
  ; use mbcnt to make sure the branch is divergent
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %cc = icmp uge i32 %hi, 16
  br i1 %cc, label %endif, label %if

if:
  %src1 = load volatile float, ptr addrspace(1) poison
  %out = fadd float %src0, %src1
  %out.0 = call float @llvm.amdgcn.wwm.f32(float %out)
  br label %endif

endif:
  %out.1 = phi float [ %out.0, %if ], [ 0.0, %main_body ]
  ret float %out.1
}

define amdgpu_ps float @test_wwm6_loop() {
main_body:
  %src0 = load volatile float, ptr addrspace(1) poison
  ; use mbcnt to make sure the branch is divergent
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  br label %loop

loop:
  %counter = phi i32 [ %hi, %main_body ], [ %counter.1, %loop ]
  %src1 = load volatile float, ptr addrspace(1) poison
  %out = fadd float %src0, %src1
  %out.0 = call float @llvm.amdgcn.wwm.f32(float %out)
  %counter.1 = sub i32 %counter, 1
  %cc = icmp ne i32 %counter.1, 0
  br i1 %cc, label %loop, label %endloop

endloop:
  ret float %out.0
}

define amdgpu_ps void @test_wwm_set_inactive1(i32 inreg %idx) {
main_body:
  %src = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx, i32 0, i32 0, i32 0)
  %src.0 = bitcast float %src to i32
  %src.1 = call i32 @llvm.amdgcn.set.inactive.i32(i32 %src.0, i32 0)
  %out = add i32 %src.1, %src.1
  %out.0 = call i32 @llvm.amdgcn.wwm.i32(i32 %out)
  %out.1 = bitcast i32 %out.0 to float
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %out.1, ptr addrspace(8) poison, i32 %idx, i32 0, i32 0, i32 0)
  ret void
}

define amdgpu_ps float @test_strict_wqm1(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx1, i32 0, i32 0, i32 0)
  %out = fadd float %src0, %src1
  %out.0 = call float @llvm.amdgcn.strict.wqm.f32(float %out)
  ret float %out.0
}

define amdgpu_ps float @test_strict_wqm2(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx1, i32 0, i32 0, i32 0)
  %src0.0 = bitcast float %src0 to i32
  %src1.0 = bitcast float %src1 to i32
  %out = add i32 %src0.0, %src1.0
  %out.0 = call i32 @llvm.amdgcn.strict.wqm.i32(i32 %out)
  %out.1 = bitcast i32 %out.0 to float
  ret float %out.1
}

define amdgpu_ps float @test_strict_wqm3(i32 inreg %idx) {
main_body:
  ; use mbcnt to make sure the branch is divergent
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %cc = icmp uge i32 %hi, 16
  br i1 %cc, label %endif, label %if

if:
  %src = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx, i32 0, i32 0, i32 0)
  %out = fadd float %src, %src
  %out.0 = call float @llvm.amdgcn.strict.wqm.f32(float %out)
  %out.1 = fadd float %src, %out.0
  br label %endif

endif:
  %out.2 = phi float [ %out.1, %if ], [ 0.0, %main_body ]
  ret float %out.2
}

define amdgpu_ps float @test_strict_wqm4(i32 inreg %idx) {
main_body:
  ; use mbcnt to make sure the branch is divergent
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %cc = icmp uge i32 %hi, 16
  br i1 %cc, label %endif, label %if

if:
  %src = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx, i32 0, i32 0, i32 0)
  %out = fadd float %src, %src
  %out.0 = call float @llvm.amdgcn.strict.wqm.f32(float %out)
  br label %endif

endif:
  %out.1 = phi float [ %out.0, %if ], [ 0.0, %main_body ]
  ret float %out.1
}

define amdgpu_ps float @test_strict_wqm5(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %src0, ptr addrspace(8) poison, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx1, i32 0, i32 0, i32 0)
  %temp = fadd float %src1, %src1
  %temp.0 = call float @llvm.amdgcn.strict.wqm.f32(float %temp)
  %out = fadd float %temp.0, %temp.0
  %out.0 = call float @llvm.amdgcn.wqm.f32(float %out)
  ret float %out.0
}

define amdgpu_ps float @test_strict_wqm6_then() {
main_body:
  %src0 = load volatile float, ptr addrspace(1) poison
  ; use mbcnt to make sure the branch is divergent
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %cc = icmp uge i32 %hi, 16
  br i1 %cc, label %endif, label %if

if:
  %src1 = load volatile float, ptr addrspace(1) poison
  %out = fadd float %src0, %src1
  %out.0 = call float @llvm.amdgcn.strict.wqm.f32(float %out)
  br label %endif

endif:
  %out.1 = phi float [ %out.0, %if ], [ 0.0, %main_body ]
  ret float %out.1
}

define amdgpu_ps float @test_strict_wqm6_loop() {
main_body:
  %src0 = load volatile float, ptr addrspace(1) poison
  ; use mbcnt to make sure the branch is divergent
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  br label %loop

loop:
  %counter = phi i32 [ %hi, %main_body ], [ %counter.1, %loop ]
  %src1 = load volatile float, ptr addrspace(1) poison
  %out = fadd float %src0, %src1
  %out.0 = call float @llvm.amdgcn.strict.wqm.f32(float %out)
  %counter.1 = sub i32 %counter, 1
  %cc = icmp ne i32 %counter.1, 0
  br i1 %cc, label %loop, label %endloop

endloop:
  ret float %out.0
}

define amdgpu_ps void @test_set_inactive2(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src1 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx1, i32 0, i32 0, i32 0)
  %src1.0 = bitcast float %src1 to i32
  %src1.1 = call i32 @llvm.amdgcn.set.inactive.i32(i32 %src1.0, i32 poison)
  %src0 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx0, i32 0, i32 0, i32 0)
  %src0.0 = bitcast float %src0 to i32
  %src0.1 = call i32 @llvm.amdgcn.wqm.i32(i32 %src0.0)
  %out = add i32 %src0.1, %src1.1
  %out.0 = bitcast i32 %out to float
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %out.0, ptr addrspace(8) poison, i32 %idx1, i32 0, i32 0, i32 0)
  ret void
}

define amdgpu_ps float @test_control_flow_0(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %c, i32 %z, float %data) {
main_body:
  %cmp = icmp eq i32 %z, 0
  br i1 %cmp, label %IF, label %ELSE

IF:
  %c.bc = bitcast i32 %c to float
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c.bc, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %data.if = extractelement <4 x float> %dtex, i32 0
  br label %END

ELSE:
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %data, ptr addrspace(8) poison, i32 %c, i32 0, i32 0, i32 0)
  br label %END

END:
  %r = phi float [ %data.if, %IF ], [ %data, %ELSE ]
  ret float %r
}

define amdgpu_ps float @test_control_flow_1(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %c, i32 %z, float %data) {
main_body:
  %cmp = icmp eq i32 %z, 0
  br i1 %cmp, label %ELSE, label %IF

IF:
  %c.bc = bitcast i32 %c to float
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c.bc, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %data.if = extractelement <4 x float> %dtex, i32 0
  br label %END

ELSE:
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %data, ptr addrspace(8) poison, i32 %c, i32 0, i32 0, i32 0)
  br label %END

END:
  %r = phi float [ %data.if, %IF ], [ %data, %ELSE ]
  ret float %r
}

define amdgpu_ps <4 x float> @test_control_flow_2(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, <3 x i32> %idx, <2 x float> %data, i32 %coord) {
main_body:
  %idx.1 = extractelement <3 x i32> %idx, i32 0
  %data.1 = extractelement <2 x float> %data, i32 0
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %data.1, ptr addrspace(8) poison, i32 %idx.1, i32 0, i32 0, i32 0)

  ; The load that determines the branch (and should therefore be WQM) is
  ; surrounded by stores that require disabled WQM.
  %idx.2 = extractelement <3 x i32> %idx, i32 1
  %z = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx.2, i32 0, i32 0, i32 0)

  %idx.3 = extractelement <3 x i32> %idx, i32 2
  %data.3 = extractelement <2 x float> %data, i32 1
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %data.3, ptr addrspace(8) poison, i32 %idx.3, i32 0, i32 0, i32 0)

  %cc = fcmp ogt float %z, 0.0
  br i1 %cc, label %IF, label %ELSE

IF:
  %coord.IF = mul i32 %coord, 3
  br label %END

ELSE:
  %coord.ELSE = mul i32 %coord, 4
  br label %END

END:
  %coord.END = phi i32 [ %coord.IF, %IF ], [ %coord.ELSE, %ELSE ]
  %coord.END.bc = bitcast i32 %coord.END to float
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord.END.bc, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  ret <4 x float> %tex
}

define amdgpu_ps float @test_control_flow_3(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %idx, float %coord) {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %dtex.1 = extractelement <4 x float> %dtex, i32 0
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %dtex.1, ptr addrspace(8) poison, i32 %idx, i32 0, i32 0, i32 0)

  %cc = fcmp ogt float %dtex.1, 0.0
  br i1 %cc, label %IF, label %ELSE

IF:
  %tex.IF = fmul float %dtex.1, 3.0
  br label %END

ELSE:
  %tex.ELSE = fmul float %dtex.1, 4.0
  br label %END

END:
  %tex.END = phi float [ %tex.IF, %IF ], [ %tex.ELSE, %ELSE ]
  ret float %tex.END
}

define amdgpu_ps <4 x float> @test_control_flow_4(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, float %coord, i32 %y, float %z) {
main_body:
  %cond = icmp eq i32 %y, 0
  br i1 %cond, label %IF, label %END

IF:
  %data = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %data, ptr addrspace(8) poison, i32 1, i32 0, i32 0, i32 0)
  br label %END

END:
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  ret <4 x float> %dtex
}

define amdgpu_ps <4 x float> @test_kill_0(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, ptr addrspace(1) inreg %ptr, <2 x i32> %idx, <2 x float> %data, float %coord, float %coord2, float %z) {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %idx.0 = extractelement <2 x i32> %idx, i32 0
  %data.0 = extractelement <2 x float> %data, i32 0
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %data.0, ptr addrspace(8) poison, i32 %idx.0, i32 0, i32 0, i32 0)

  %z.cmp = fcmp olt float %z, 0.0
  call void @llvm.amdgcn.kill(i1 %z.cmp)

  %idx.1 = extractelement <2 x i32> %idx, i32 1
  %data.1 = extractelement <2 x float> %data, i32 1
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %data.1, ptr addrspace(8) poison, i32 %idx.1, i32 0, i32 0, i32 0)
  %tex2 = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord2, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %tex2.0 = extractelement <4 x float> %tex2, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex2.0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %out = fadd <4 x float> %tex, %dtex

  ret <4 x float> %out
}

define amdgpu_ps <4 x float> @test_kill_1(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %idx, float %data, float %coord, float %coord2, float %z) {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0

  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %data, ptr addrspace(8) poison, i32 0, i32 0, i32 0)

  %z.cmp = fcmp olt float %z, 0.0
  call void @llvm.amdgcn.kill(i1 %z.cmp)

  ret <4 x float> %dtex
}

define amdgpu_ps float @test_prolog_1(float %a, float %b) #5 {
main_body:
  %s = fadd float %a, %b
  ret float %s
}

define amdgpu_ps <4 x float> @test_loop_vcc(<4 x float> %in) nounwind {
entry:
  call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %in, i32 15, i32 poison, <8 x i32> poison, i32 0, i32 0)
  br label %loop

loop:
  %ctr.iv = phi float [ 0.0, %entry ], [ %ctr.next, %body ]
  %c.iv = phi <4 x float> [ %in, %entry ], [ %c.next, %body ]
  %cc = fcmp ogt float %ctr.iv, 7.0
  br i1 %cc, label %break, label %body

body:
  %c.iv0 = extractelement <4 x float> %c.iv, i32 0
  %c.next = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c.iv0, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0) #0
  %ctr.next = fadd float %ctr.iv, 2.0
  br label %loop

break:
  ret <4 x float> %c.iv
}

define amdgpu_ps void @test_alloca(float %data, i32 %a, i32 %idx) nounwind {
entry:
  %array = alloca [32 x i32], align 4, addrspace(5)

  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %data, ptr addrspace(8) poison, i32 0, i32 0, i32 0)

  store volatile i32 %a, ptr addrspace(5) %array, align 4

  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %data, ptr addrspace(8) poison, i32 1, i32 0, i32 0, i32 0)

  %c.gep = getelementptr [32 x i32], ptr addrspace(5) %array, i32 0, i32 %idx
  %c = load i32, ptr addrspace(5) %c.gep, align 4
  %c.bc = bitcast i32 %c to float
  %t = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c.bc, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0) #0
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4f32(<4 x float> %t, ptr addrspace(8) poison, i32 0, i32 0, i32 0)

  ret void
}

define amdgpu_ps <4 x float> @test_nonvoid_return() nounwind {
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0) #0
  ret <4 x float> %dtex
}

define amdgpu_ps <4 x float> @test_nonvoid_return_unreachable(i32 inreg %c) nounwind {
entry:
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0) #0
  %cc = icmp sgt i32 %c, 0
  br i1 %cc, label %if, label %else

if:
  store volatile <4 x float> %dtex, ptr addrspace(1) poison
  unreachable

else:
  ret <4 x float> %dtex
}

define amdgpu_ps <4 x float> @test_scc(i32 inreg %sel, i32 %idx) #1 {
main_body:
  %cc = icmp sgt i32 %sel, 0
  br i1 %cc, label %if, label %else

if:
  %r.if = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float 0.0, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0) #0
  br label %end

else:
  %r.else = call <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 15, float 0.0, float bitcast (i32 1 to float), <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0) #0
  br label %end

end:
  %r = phi <4 x float> [ %r.if, %if ], [ %r.else, %else ]
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float 1.0, ptr addrspace(8) poison, i32 %idx, i32 0, i32 0, i32 0)
  ret <4 x float> %r
}

define amdgpu_ps float @test_wwm_within_wqm(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %c, i32 %z, float %data) {
main_body:
  %cmp = icmp eq i32 %z, 0
  br i1 %cmp, label %IF, label %ENDIF

IF:
  %c.bc = bitcast i32 %c to float
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c.bc, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %dataf = extractelement <4 x float> %dtex, i32 0
  %data1 = fptosi float %dataf to i32
  %data2 = call i32 @llvm.amdgcn.set.inactive.i32(i32 %data1, i32 0)
  %data3 = call i32 @llvm.amdgcn.ds.swizzle(i32 %data2, i32 2079)
  %data4 = call i32 @llvm.amdgcn.wwm.i32(i32 %data3)
  %data4f = sitofp i32 %data4 to float
  br label %ENDIF

ENDIF:
  %r = phi float [ %data4f, %IF ], [ 0.0, %main_body ]
  ret float %r
}

define amdgpu_ps float @test_strict_wwm1(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx1, i32 0, i32 0, i32 0)
  %out = fadd float %src0, %src1
  %out.0 = call float @llvm.amdgcn.strict.wwm.f32(float %out)
  ret float %out.0
}

define amdgpu_ps float @test_strict_wwm2(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx1, i32 0, i32 0, i32 0)
  %src0.0 = bitcast float %src0 to i32
  %src1.0 = bitcast float %src1 to i32
  %out = add i32 %src0.0, %src1.0
  %out.0 = call i32 @llvm.amdgcn.strict.wwm.i32(i32 %out)
  %out.1 = bitcast i32 %out.0 to float
  ret float %out.1
}

define amdgpu_ps float @test_strict_wwm3(i32 inreg %idx) {
main_body:
  ; use mbcnt to make sure the branch is divergent
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %cc = icmp uge i32 %hi, 16
  br i1 %cc, label %endif, label %if

if:
  %src = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx, i32 0, i32 0, i32 0)
  %out = fadd float %src, %src
  %out.0 = call float @llvm.amdgcn.strict.wwm.f32(float %out)
  %out.1 = fadd float %src, %out.0
  br label %endif

endif:
  %out.2 = phi float [ %out.1, %if ], [ 0.0, %main_body ]
  ret float %out.2
}

define amdgpu_ps float @test_strict_wwm4(i32 inreg %idx) {
main_body:
  ; use mbcnt to make sure the branch is divergent
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %cc = icmp uge i32 %hi, 16
  br i1 %cc, label %endif, label %if

if:
  %src = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx, i32 0, i32 0, i32 0)
  %out = fadd float %src, %src
  %out.0 = call float @llvm.amdgcn.strict.wwm.f32(float %out)
  br label %endif

endif:
  %out.1 = phi float [ %out.0, %if ], [ 0.0, %main_body ]
  ret float %out.1
}

define amdgpu_ps float @test_strict_wwm5(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %src0, ptr addrspace(8) poison, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx1, i32 0, i32 0, i32 0)
  %temp = fadd float %src1, %src1
  %temp.0 = call float @llvm.amdgcn.strict.wwm.f32(float %temp)
  %out = fadd float %temp.0, %temp.0
  %out.0 = call float @llvm.amdgcn.wqm.f32(float %out)
  ret float %out.0
}

define amdgpu_ps float @test_strict_wwm6_then() {
main_body:
  %src0 = load volatile float, ptr addrspace(1) poison
  ; use mbcnt to make sure the branch is divergent
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %cc = icmp uge i32 %hi, 16
  br i1 %cc, label %endif, label %if

if:
  %src1 = load volatile float, ptr addrspace(1) poison
  %out = fadd float %src0, %src1
  %out.0 = call float @llvm.amdgcn.strict.wwm.f32(float %out)
  br label %endif

endif:
  %out.1 = phi float [ %out.0, %if ], [ 0.0, %main_body ]
  ret float %out.1
}

define amdgpu_ps float @test_strict_wwm6_loop() {
main_body:
  %src0 = load volatile float, ptr addrspace(1) poison
  ; use mbcnt to make sure the branch is divergent
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  br label %loop

loop:
  %counter = phi i32 [ %hi, %main_body ], [ %counter.1, %loop ]
  %src1 = load volatile float, ptr addrspace(1) poison
  %out = fadd float %src0, %src1
  %out.0 = call float @llvm.amdgcn.strict.wwm.f32(float %out)
  %counter.1 = sub i32 %counter, 1
  %cc = icmp ne i32 %counter.1, 0
  br i1 %cc, label %loop, label %endloop

endloop:
  ret float %out.0
}

define amdgpu_ps void @test_strict_wwm_set_inactive1(i32 inreg %idx) {
main_body:
  %src = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) poison, i32 %idx, i32 0, i32 0, i32 0)
  %src.0 = bitcast float %src to i32
  %src.1 = call i32 @llvm.amdgcn.set.inactive.i32(i32 %src.0, i32 0)
  %out = add i32 %src.1, %src.1
  %out.0 = call i32 @llvm.amdgcn.strict.wwm.i32(i32 %out)
  %out.1 = bitcast i32 %out.0 to float
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %out.1, ptr addrspace(8) poison, i32 %idx, i32 0, i32 0, i32 0)
  ret void
}

define amdgpu_ps float @test_strict_wwm_within_wqm(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %c, i32 %z, float %data) {
main_body:
  %cmp = icmp eq i32 %z, 0
  br i1 %cmp, label %IF, label %ENDIF

IF:
  %c.bc = bitcast i32 %c to float
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c.bc, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %dataf = extractelement <4 x float> %dtex, i32 0
  %data1 = fptosi float %dataf to i32
  %data2 = call i32 @llvm.amdgcn.set.inactive.i32(i32 %data1, i32 0)
  %data3 = call i32 @llvm.amdgcn.ds.swizzle(i32 %data2, i32 2079)
  %data4 = call i32 @llvm.amdgcn.strict.wwm.i32(i32 %data3)
  %data4f = sitofp i32 %data4 to float
  br label %ENDIF

ENDIF:
  %r = phi float [ %data4f, %IF ], [ 0.0, %main_body ]
  ret float %r
}

define amdgpu_ps float @test_strict_wqm_within_wqm(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %c, i32 %z, float %data) {
main_body:
  %cmp = icmp eq i32 %z, 0
  br i1 %cmp, label %IF, label %ENDIF

IF:
  %c.bc = bitcast i32 %c to float
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c.bc, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %dataf = extractelement <4 x float> %dtex, i32 0
  %data1 = fptosi float %dataf to i32
  %data2 = call i32 @llvm.amdgcn.ds.swizzle(i32 %data1, i32 2079)
  %data3 = call i32 @llvm.amdgcn.strict.wqm.i32(i32 %data2)
  %data3f = sitofp i32 %data3 to float
  br label %ENDIF

ENDIF:
  %r = phi float [ %data3f, %IF ], [ 0.0, %main_body ]
  ret float %r
}

define amdgpu_ps float @test_strict_wqm_within_wqm_with_kill(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %c, i32 %z, float %data, i32 %wqm_data) {
main_body:
  %c.bc = bitcast i32 %c to float
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c.bc, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %cmp = icmp eq i32 %z, 0
  call void @llvm.amdgcn.kill(i1 %cmp)
  %dataf = extractelement <4 x float> %dtex, i32 0
  %data2 = call i32 @llvm.amdgcn.ds.swizzle(i32 %wqm_data, i32 2079)
  %data3 = call i32 @llvm.amdgcn.strict.wqm.i32(i32 %data2)
  %data3f = sitofp i32 %data3 to float
  %result.f = fadd float %dataf, %data3f
  %result.i = bitcast float %result.f to i32
  %result.wqm = call i32 @llvm.amdgcn.wqm.i32(i32 %result.i)
  %result = bitcast i32 %result.wqm to float
  ret float %result
}

define amdgpu_ps float @test_strict_wqm_strict_wwm_wqm(i32 inreg %idx0, i32 inreg %idx1, ptr addrspace(8) inreg %res, ptr addrspace(8) inreg %res2, float %inp, <8 x i32> inreg %res3) {
main_body:
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %inp, ptr addrspace(8) %res, i32 %idx1, i32 0, i32 0, i32 0)
  %reload = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) %res, i32 %idx1, i32 0, i32 0, i32 0)
  %temp = fadd float %reload, %reload
  %temp2 = call float @llvm.amdgcn.strict.wqm.f32(float %temp)
  %temp3 = fadd float %temp2, %temp2
  %reload_wwm = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) %res2, i32 %idx0, i32 0, i32 0, i32 0)
  %temp4 = call float @llvm.amdgcn.strict.wwm.f32(float %reload_wwm)
  %temp5 = fadd float %temp3, %temp4
  %res.int = ptrtoint ptr addrspace(8) %res to i128
  %res.vec = bitcast i128 %res.int to <4 x i32>
  %tex = call float @llvm.amdgcn.image.sample.1d.f32.f32(i32 1, float %temp5, <8 x i32> %res3, <4 x i32> %res.vec, i1 false, i32 0, i32 0)
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %tex, ptr addrspace(8) %res, i32 %idx1, i32 0, i32 0, i32 0)
  %out = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) %res, i32 %idx1, i32 0, i32 0, i32 0)
  ret float %out
}

define amdgpu_ps float @test_strict_wwm_strict_wqm_wqm(i32 inreg %idx0, i32 inreg %idx1, ptr addrspace(8) inreg %res, float %inp, <8 x i32> inreg %res2) {
main_body:
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %inp, ptr addrspace(8) %res, i32 %idx0, i32 0, i32 0, i32 0)
  %reload = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) %res, i32 %idx1, i32 0, i32 0, i32 0)
  %temp = fadd float %reload, %reload
  %temp2 = call float @llvm.amdgcn.strict.wwm.f32(float %temp)
  %temp3 = fadd float %temp2, %temp2
  %reload_wwm = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) %res, i32 %idx0, i32 0, i32 0, i32 0)
  %temp4 = call float @llvm.amdgcn.strict.wqm.f32(float %reload_wwm)
  %temp5 = fadd float %temp3, %temp4
  %res.int = ptrtoint ptr addrspace(8) %res to i128
  %res.vec = bitcast i128 %res.int to <4 x i32>
  %tex = call float @llvm.amdgcn.image.sample.1d.f32.f32(i32 1, float %temp5, <8 x i32> %res2, <4 x i32> %res.vec, i1 false, i32 0, i32 0)
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %tex, ptr addrspace(8) %res, i32 %idx0, i32 0, i32 0, i32 0)
  %out = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) %res, i32 %idx0, i32 0, i32 0, i32 0)
  ret float %out
}

define amdgpu_ps float @test_wqm_strict_wqm_wqm(i32 inreg %idx0, i32 inreg %idx1, ptr addrspace(8) inreg %res, float %inp, <8 x i32> inreg %res2) {
main_body:
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %inp, ptr addrspace(8) %res, i32 %idx0, i32 0, i32 0, i32 0)
  %reload = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) %res, i32 %idx1, i32 0, i32 0, i32 0)
  %temp = fadd float %reload, %reload
  %res.int = ptrtoint ptr addrspace(8) %res to i128
  %res.vec = bitcast i128 %res.int to <4 x i32>
  %tex = call float @llvm.amdgcn.image.sample.1d.f32.f32(i32 1, float %temp, <8 x i32> %res2, <4 x i32> %res.vec, i1 false, i32 0, i32 0)
  %temp2 = fadd float %tex, %tex
  %reload_wwm = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) %res, i32 %idx0, i32 0, i32 0, i32 0)
  %temp3 = call float @llvm.amdgcn.strict.wqm.f32(float %reload_wwm)
  %temp4 = fadd float %temp2, %temp3
  %tex2 = call float @llvm.amdgcn.image.sample.1d.f32.f32(i32 1, float %temp4, <8 x i32> %res2, <4 x i32> %res.vec, i1 false, i32 0, i32 0)
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %tex2, ptr addrspace(8) %res, i32 %idx0, i32 0, i32 0, i32 0)
  %out = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) %res, i32 %idx0, i32 0, i32 0, i32 0)
  ret float %out
}

define amdgpu_ps void @test_for_deactivating_lanes_in_wave32(ptr addrspace(6) inreg %0) {
main_body:
  %1 = ptrtoint ptr addrspace(6) %0 to i32
  %2 = insertelement <4 x i32> <i32 poison, i32 32768, i32 32, i32 822177708>, i32 %1, i32 0
  %3 = call nsz arcp float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %2, i32 0, i32 0) #3
  %4 = fcmp nsz arcp ugt float %3, 0.000000e+00
  call void @llvm.amdgcn.kill(i1 %4) #1
  ret void
}

define amdgpu_gs void @wqm_init_exec() {
bb:
  call void @llvm.amdgcn.init.exec(i64 -1)
  call void @llvm.amdgcn.raw.buffer.store.v4f32(<4 x float> zeroinitializer, <4 x i32> zeroinitializer, i32 0, i32 0, i32 0)
  %i = call i32 @llvm.amdgcn.wqm.i32(i32 0)
  store i32 %i, i32 addrspace(3)* null, align 4
  ret void
}

define amdgpu_gs void @wqm_init_exec_switch(i32 %arg) {
  call void @llvm.amdgcn.init.exec(i64 0)
  switch i32 %arg, label %bb1 [
    i32 0, label %bb3
    i32 1, label %bb2
  ]
bb1:
  ret void
bb2:
  ret void
bb3:
  ret void
}

define amdgpu_gs void @wqm_init_exec_wwm() {
  call void @llvm.amdgcn.init.exec(i64 0)
  %i = call i64 @llvm.amdgcn.ballot.i64(i1 true)
  %i1 = call i32 @llvm.amdgcn.wwm.i32(i32 0)
  %i2 = insertelement <2 x i32> zeroinitializer, i32 %i1, i64 0
  %i3 = bitcast <2 x i32> %i2 to i64
  %i4 = icmp ne i64 %i, 0
  %i5 = icmp ne i64 %i3, 0
  %i6 = xor i1 %i4, %i5
  %i7 = uitofp i1 %i6 to float
  call void @llvm.amdgcn.exp.f32(i32 0, i32 0, float %i7, float 0.0, float 0.0, float 0.0, i1 false, i1 false)
  ret void
}

define amdgpu_ps float @short_exact_regions(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, float %c, ptr addrspace(4) %p) {
main_body:
  %tex1 = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %idx0 = load <4 x i32>, ptr addrspace(4) %p, align 4
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %cc = icmp uge i32 %hi, 16
  br i1 %cc, label %endif, label %if

if:
  %idx1 = extractelement <4 x i32> %idx0, i64 0
  %idx2 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %idx1)
  %idx3 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %sampler, i32 %idx2, i32 0)

  call void @llvm.amdgcn.struct.buffer.store.v4f32(<4 x float> %tex1, <4 x i32> poison, i32 %idx3, i32 0, i32 0, i32 0)
  br label %endif

endif:
  %d = extractelement <4 x float> %tex1, i64 0
  %tex2 = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %d, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %r0 = extractelement <4 x float> %tex1, i64 1
  %r1 = extractelement <4 x float> %tex2, i64 2
  %r2 = fadd float %r0, %r1
  %out = call float @llvm.amdgcn.wqm.f32(float %r2)

  ret float %out
}

define amdgpu_ps float @short_exact_regions_2(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, float %c, ptr addrspace(4) %p) {
main_body:
  %tex1 = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0
  %idx0 = load <4 x i32>, ptr addrspace(4) %p, align 4
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %idx1 = extractelement <4 x i32> %idx0, i64 0
  %d = extractelement <4 x float> %tex1, i64 0

  %tex2 = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %d, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0) #0

  %idx2 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %idx1)
  %idx3 = call float @llvm.amdgcn.s.buffer.load.f32(<4 x i32> %sampler, i32 %idx2, i32 0)

  %r0 = extractelement <4 x float> %tex1, i64 1
  %r1 = extractelement <4 x float> %tex2, i64 2
  %r2 = fadd float %r0, %r1
  %out = fadd float %r2, %idx3

  ret float %out
}

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #1
declare void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float>, i32, i32, <8 x i32>, i32, i32) #1

declare void @llvm.amdgcn.struct.buffer.store.f32(float, <4 x i32>, i32, i32, i32, i32 immarg) #2
declare void @llvm.amdgcn.struct.buffer.store.v4f32(<4 x float>, <4 x i32>, i32, i32, i32, i32 immarg) #2
declare void @llvm.amdgcn.raw.buffer.store.v4f32(<4 x float>, <4 x i32>, i32, i32, i32 immarg) #2
declare void @llvm.amdgcn.raw.buffer.store.f32(float, <4 x i32>, i32, i32, i32 immarg) #2
declare float @llvm.amdgcn.raw.buffer.load.f32(<4 x i32>, i32, i32, i32) #3
declare float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32>, i32, i32, i32, i32) #3

declare void @llvm.amdgcn.struct.ptr.buffer.store.f32(float, ptr addrspace(8), i32, i32, i32, i32 immarg) #2
declare void @llvm.amdgcn.struct.ptr.buffer.store.v4f32(<4 x float>, ptr addrspace(8), i32, i32, i32, i32 immarg) #2
declare void @llvm.amdgcn.raw.ptr.buffer.store.v4f32(<4 x float>, ptr addrspace(8), i32, i32, i32 immarg) #2
declare void @llvm.amdgcn.raw.ptr.buffer.store.f32(float, ptr addrspace(8), i32, i32, i32 immarg) #2
declare float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8), i32, i32, i32) #3
declare float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8), i32, i32, i32, i32) #3

declare <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32, i32, <8 x i32>, i32, i32) #3
declare <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #3
declare <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #3
declare float @llvm.amdgcn.image.sample.1d.f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #3
declare void @llvm.amdgcn.kill(i1) #1
declare float @llvm.amdgcn.wqm.f32(float) #3
declare i32 @llvm.amdgcn.wqm.i32(i32) #3
declare float @llvm.amdgcn.strict.wwm.f32(float) #3
declare i32 @llvm.amdgcn.strict.wwm.i32(i32) #3
declare float @llvm.amdgcn.wwm.f32(float) #3
declare i32 @llvm.amdgcn.wwm.i32(i32) #3
declare float @llvm.amdgcn.strict.wqm.f32(float) #3
declare i32 @llvm.amdgcn.strict.wqm.i32(i32) #3
declare i32 @llvm.amdgcn.set.inactive.i32(i32, i32) #4
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #3
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #3
declare <2 x half> @llvm.amdgcn.cvt.pkrtz(float, float) #3
declare void @llvm.amdgcn.exp.compr.v2f16(i32, i32, <2 x half>, <2 x half>, i1, i1) #1
declare float @llvm.amdgcn.interp.p1(float, i32, i32, i32) #2
declare float @llvm.amdgcn.interp.p2(float, float, i32, i32, i32) #2
declare i32 @llvm.amdgcn.ds.swizzle(i32, i32)
declare float @llvm.amdgcn.s.buffer.load.f32(<4 x i32>, i32, i32 immarg) #7
declare i32 @llvm.amdgcn.readfirstlane.i32(i32)

attributes #1 = { nounwind }
attributes #2 = { nounwind readonly }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind readnone convergent }
attributes #5 = { "amdgpu-ps-wqm-outputs" }
attributes #6 = { nounwind "InitialPSInputAddr"="2" }
attributes #7 = { nounwind readnone willreturn }
