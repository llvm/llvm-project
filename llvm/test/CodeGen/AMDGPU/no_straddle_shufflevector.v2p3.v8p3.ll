; Test --amdgpu-prevent-half-cache-line-straddling with MetaInstructions.
; Based on shufflevector.v2p3.v8p3.ll

;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900  -mattr=dumpcode --filetype=obj < %s | llvm-objdump --triple=amdgcn-amd-amdhsa --mcpu=gfx900 -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis

;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a  -mattr=dumpcode --filetype=obj < %s | llvm-objdump --triple=amdgcn-amd-amdhsa --mcpu=gfx90a -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis

;RUN: llc --amdgpu-prevent-half-cache-line-straddling -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942  -mattr=dumpcode --filetype=obj < %s | llvm-objdump --triple=amdgcn-amd-amdhsa --mcpu=gfx942 -d  - > %t.dis
;RUN: %python %p/has_cache_straddle.py %t.dis

define void @v_shuffle_v2p3_v8p3__u_u(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> poison
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__0_u(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 0, i32 poison>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__1_u(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 poison>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__2_u(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 poison>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__3_u(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 poison>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__4_u(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 poison>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__5_u(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 poison>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__6_u(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 poison>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__7_u(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 poison>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__8_u(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 poison>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__9_u(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 poison>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__10_u(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 poison>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__11_u(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 poison>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__12_u(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 poison>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__13_u(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 poison>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__14_u(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 poison>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__15_u(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 poison>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__15_0(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 0>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__15_1(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 1>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__15_2(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 2>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__15_3(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 3>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__15_4(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 4>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__15_5(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 5>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__15_6(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 6>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__15_7(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 7>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__15_8(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 8>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__15_9(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 9>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__15_10(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 10>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__15_11(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 11>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__15_12(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 12>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__15_13(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 13>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__15_14(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 14>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__15_15(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 15>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__u_0(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 poison, i32 0>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__0_0(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> zeroinitializer
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__1_0(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 0>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__2_0(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 0>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__3_0(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 0>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__4_0(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 0>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__5_0(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 0>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__6_0(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 0>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__7_0(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 0>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__8_0(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 0>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__9_0(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 0>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__10_0(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 0>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__11_0(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 0>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__12_0(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 0>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__13_0(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 0>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__14_0(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 0>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__u_1(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 poison, i32 1>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__0_1(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 0, i32 1>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__1_1(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 1>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__2_1(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 1>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__3_1(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 1>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__4_1(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 1>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__5_1(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 1>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__6_1(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 1>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__7_1(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 1>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__8_1(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 1>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__9_1(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 1>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__10_1(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 1>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__11_1(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 1>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__12_1(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 1>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__13_1(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 1>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__14_1(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 1>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__u_2(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 poison, i32 2>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__0_2(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 0, i32 2>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__1_2(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 2>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__2_2(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 2>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__3_2(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 2>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__4_2(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 2>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__5_2(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 2>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__6_2(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 2>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__7_2(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 2>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__8_2(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 2>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__9_2(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 2>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__10_2(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 2>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__11_2(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 2>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__12_2(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 2>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__13_2(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 2>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__14_2(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 2>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__u_3(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 poison, i32 3>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__0_3(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 0, i32 3>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__1_3(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 3>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__2_3(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 3>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__3_3(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 3>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__4_3(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 3>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__5_3(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 3>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__6_3(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 3>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__7_3(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 3>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__8_3(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 3>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__9_3(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 3>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__10_3(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 3>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__11_3(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 3>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__12_3(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 3>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__13_3(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 3>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__14_3(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 3>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__u_4(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 poison, i32 4>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__0_4(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 0, i32 4>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__1_4(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 4>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__2_4(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 4>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__3_4(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 4>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__4_4(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 4>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__5_4(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 4>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__6_4(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 4>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__7_4(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 4>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__8_4(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 4>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__9_4(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 4>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__10_4(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 4>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__11_4(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 4>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__12_4(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 4>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__13_4(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 4>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__14_4(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 4>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__u_5(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 poison, i32 5>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__0_5(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 0, i32 5>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__1_5(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 5>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__2_5(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 5>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__3_5(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 5>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__4_5(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 5>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__5_5(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 5>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__6_5(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 5>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__7_5(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 5>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__8_5(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 5>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__9_5(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 5>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__10_5(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 5>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__11_5(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 5>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__12_5(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 5>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__13_5(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 5>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__14_5(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 5>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__u_6(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 poison, i32 6>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__0_6(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 0, i32 6>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__1_6(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 6>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__2_6(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 6>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__3_6(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 6>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__4_6(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 6>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__5_6(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 6>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__6_6(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 6>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__7_6(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 6>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__8_6(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 6>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__9_6(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 6>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__10_6(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 6>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__11_6(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 6>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__12_6(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 6>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__13_6(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 6>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__14_6(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 6>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__u_7(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 poison, i32 7>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__0_7(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 0, i32 7>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__1_7(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 7>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__2_7(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 7>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__3_7(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 7>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__4_7(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 7>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__5_7(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 7>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__6_7(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 7>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__7_7(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 7>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__8_7(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 7>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__9_7(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 7>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__10_7(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 7>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__11_7(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 7>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__12_7(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 7>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__13_7(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 7>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__14_7(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 7>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__u_8(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 poison, i32 8>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__0_8(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 0, i32 8>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__1_8(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 8>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__2_8(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 8>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__3_8(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 8>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__4_8(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 8>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__5_8(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 8>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__6_8(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 8>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__7_8(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 8>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__8_8(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 8>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__9_8(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 8>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__10_8(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 8>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__11_8(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 8>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__12_8(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 8>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__13_8(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 8>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__14_8(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 8>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__u_9(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 poison, i32 9>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__0_9(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 0, i32 9>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__1_9(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 1, i32 9>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__2_9(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 2, i32 9>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__3_9(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 3, i32 9>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__4_9(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 4, i32 9>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__5_9(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 5, i32 9>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__6_9(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 6, i32 9>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__7_9(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 7, i32 9>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__8_9(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 8, i32 9>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__9_9(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 9>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__10_9(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 9>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__11_9(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 9>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__12_9(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 9>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__13_9(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 9>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__14_9(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 9>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__u_10(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 poison, i32 10>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__0_10(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 0, i32 10>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__1_10(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 1, i32 10>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__2_10(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 2, i32 10>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__3_10(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 3, i32 10>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__4_10(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 4, i32 10>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__5_10(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 5, i32 10>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__6_10(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 6, i32 10>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__7_10(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 7, i32 10>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__8_10(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 8, i32 10>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__9_10(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 10>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__10_10(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 10>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__11_10(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 10>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__12_10(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 10>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__13_10(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 10>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__14_10(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 10>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__u_11(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 poison, i32 11>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__0_11(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 0, i32 11>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__1_11(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 1, i32 11>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__2_11(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 2, i32 11>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__3_11(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 3, i32 11>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__4_11(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 4, i32 11>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__5_11(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 5, i32 11>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__6_11(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 6, i32 11>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__7_11(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 7, i32 11>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__8_11(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 8, i32 11>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__9_11(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 11>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__10_11(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 11>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__11_11(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 11>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__12_11(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 11>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__13_11(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 11>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__14_11(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 11>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__u_12(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 poison, i32 12>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__0_12(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 0, i32 12>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__1_12(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 1, i32 12>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__2_12(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 2, i32 12>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__3_12(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 3, i32 12>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__4_12(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 4, i32 12>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__5_12(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 5, i32 12>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__6_12(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 6, i32 12>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__7_12(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 7, i32 12>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__8_12(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 8, i32 12>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__9_12(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 12>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__10_12(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 12>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__11_12(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 12>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__12_12(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 12>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__13_12(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 12>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__14_12(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 12>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__u_13(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 poison, i32 13>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__0_13(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 0, i32 13>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__1_13(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 1, i32 13>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__2_13(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 2, i32 13>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__3_13(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 3, i32 13>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__4_13(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 4, i32 13>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__5_13(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 5, i32 13>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__6_13(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 6, i32 13>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__7_13(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 7, i32 13>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__8_13(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 8, i32 13>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__9_13(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 13>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__10_13(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 13>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__11_13(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 13>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__12_13(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 13>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__13_13(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 13>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__14_13(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 13>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__u_14(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 poison, i32 14>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__0_14(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 0, i32 14>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__1_14(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 1, i32 14>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__2_14(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 2, i32 14>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__3_14(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 3, i32 14>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__4_14(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 4, i32 14>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__5_14(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 5, i32 14>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__6_14(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 6, i32 14>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__7_14(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 7, i32 14>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__8_14(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 8, i32 14>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__9_14(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 14>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__10_14(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 14>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__11_14(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 14>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__12_14(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 14>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__13_14(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 14>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__14_14(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 14>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__u_15(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 poison, i32 15>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__0_15(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 0, i32 15>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__1_15(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 1, i32 15>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__2_15(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 2, i32 15>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__3_15(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 3, i32 15>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__4_15(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 4, i32 15>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__5_15(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 5, i32 15>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__6_15(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 6, i32 15>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__7_15(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 7, i32 15>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__8_15(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 8, i32 15>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__9_15(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 15>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__10_15(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 15>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__11_15(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 15>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__12_15(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 15>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__13_15(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 15>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @v_shuffle_v2p3_v8p3__14_15(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=v"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 15>
  store <2 x ptr addrspace(3)> %shuf, ptr addrspace(1) %ptr, align 8
  ret void
}

define void @s_shuffle_v2p3_v8p3__u_u() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> poison
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__0_u() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 0, i32 poison>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__1_u() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 poison>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__2_u() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 poison>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__3_u() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 poison>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__4_u() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 poison>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__5_u() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 poison>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__6_u() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 poison>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__7_u() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 poison>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__8_u() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 poison>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__9_u() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 poison>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__10_u() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 poison>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__11_u() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 poison>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__12_u() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 poison>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__13_u() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 poison>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__14_u() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 poison>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__15_u() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 poison>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__15_0() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 0>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__15_1() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 1>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__15_2() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 2>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__15_3() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 3>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__15_4() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 4>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__15_5() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 5>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__15_6() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 6>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__15_7() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 7>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__15_8() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 8>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__15_9() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 9>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__15_10() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 10>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__15_11() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 11>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__15_12() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 12>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__15_13() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 13>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__15_14() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 14>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__15_15() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 15, i32 15>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__u_0() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 poison, i32 0>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__0_0() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> zeroinitializer
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__1_0() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 0>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__2_0() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 0>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__3_0() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 0>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__4_0() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 0>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__5_0() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 0>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__6_0() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 0>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__7_0() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 0>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__8_0() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 0>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__9_0() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 0>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__10_0() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 0>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__11_0() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 0>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__12_0() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 0>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__13_0() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 0>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__14_0() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 0>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__u_1() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 poison, i32 1>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__0_1() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 0, i32 1>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__1_1() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 1>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__2_1() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 1>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__3_1() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 1>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__4_1() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 1>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__5_1() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 1>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__6_1() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 1>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__7_1() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 1>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__8_1() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 1>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__9_1() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 1>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__10_1() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 1>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__11_1() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 1>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__12_1() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 1>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__13_1() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 1>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__14_1() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 1>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__u_2() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 poison, i32 2>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__0_2() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 0, i32 2>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__1_2() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 2>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__2_2() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 2>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__3_2() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 2>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__4_2() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 2>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__5_2() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 2>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__6_2() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 2>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__7_2() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 2>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__8_2() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 2>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__9_2() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 2>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__10_2() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 2>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__11_2() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 2>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__12_2() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 2>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__13_2() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 2>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__14_2() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 2>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__u_3() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 poison, i32 3>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__0_3() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 0, i32 3>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__1_3() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 3>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__2_3() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 3>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__3_3() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 3>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__4_3() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 3>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__5_3() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 3>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__6_3() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 3>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__7_3() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 3>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__8_3() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 3>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__9_3() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 3>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__10_3() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 3>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__11_3() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 3>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__12_3() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 3>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__13_3() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 3>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__14_3() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 3>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__u_4() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 poison, i32 4>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__0_4() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 0, i32 4>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__1_4() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 4>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__2_4() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 4>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__3_4() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 4>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__4_4() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 4>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__5_4() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 4>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__6_4() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 4>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__7_4() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 4>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__8_4() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 4>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__9_4() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 4>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__10_4() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 4>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__11_4() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 4>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__12_4() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 4>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__13_4() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 4>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__14_4() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 4>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__u_5() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 poison, i32 5>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__0_5() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 0, i32 5>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__1_5() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 5>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__2_5() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 5>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__3_5() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 5>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__4_5() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 5>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__5_5() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 5>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__6_5() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 5>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__7_5() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 5>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__8_5() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 5>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__9_5() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 5>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__10_5() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 5>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__11_5() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 5>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__12_5() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 5>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__13_5() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 5>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__14_5() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 5>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__u_6() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 poison, i32 6>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__0_6() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 0, i32 6>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__1_6() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 6>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__2_6() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 6>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__3_6() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 6>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__4_6() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 6>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__5_6() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 6>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__6_6() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 6>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__7_6() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 6>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__8_6() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 6>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__9_6() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 6>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__10_6() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 6>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__11_6() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 6>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__12_6() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 6>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__13_6() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 6>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__14_6() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 6>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__u_7() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 poison, i32 7>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__0_7() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 0, i32 7>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__1_7() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 7>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__2_7() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 7>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__3_7() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 7>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__4_7() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 7>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__5_7() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 7>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__6_7() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 7>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__7_7() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 7>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__8_7() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 7>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__9_7() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 7>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__10_7() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 7>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__11_7() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 7>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__12_7() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 7>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__13_7() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 7>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__14_7() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 7>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__u_8() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 poison, i32 8>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__0_8() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 0, i32 8>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__1_8() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 1, i32 8>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__2_8() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 2, i32 8>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__3_8() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 3, i32 8>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__4_8() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 4, i32 8>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__5_8() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 5, i32 8>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__6_8() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 6, i32 8>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__7_8() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 7, i32 8>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__8_8() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> poison, <2 x i32> <i32 8, i32 8>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__9_8() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 8>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__10_8() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 8>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__11_8() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 8>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__12_8() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 8>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__13_8() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 8>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__14_8() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 8>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__u_9() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 poison, i32 9>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__0_9() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 0, i32 9>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__1_9() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 1, i32 9>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__2_9() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 2, i32 9>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__3_9() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 3, i32 9>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__4_9() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 4, i32 9>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__5_9() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 5, i32 9>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__6_9() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 6, i32 9>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__7_9() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 7, i32 9>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__8_9() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 8, i32 9>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__9_9() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 9>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__10_9() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 9>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__11_9() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 9>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__12_9() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 9>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__13_9() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 9>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__14_9() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 9>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__u_10() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 poison, i32 10>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__0_10() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 0, i32 10>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__1_10() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 1, i32 10>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__2_10() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 2, i32 10>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__3_10() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 3, i32 10>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__4_10() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 4, i32 10>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__5_10() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 5, i32 10>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__6_10() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 6, i32 10>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__7_10() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 7, i32 10>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__8_10() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 8, i32 10>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__9_10() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 10>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__10_10() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 10>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__11_10() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 10>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__12_10() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 10>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__13_10() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 10>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__14_10() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 10>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__u_11() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 poison, i32 11>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__0_11() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 0, i32 11>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__1_11() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 1, i32 11>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__2_11() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 2, i32 11>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__3_11() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 3, i32 11>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__4_11() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 4, i32 11>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__5_11() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 5, i32 11>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__6_11() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 6, i32 11>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__7_11() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 7, i32 11>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__8_11() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 8, i32 11>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__9_11() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 11>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__10_11() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 11>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__11_11() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 11>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__12_11() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 11>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__13_11() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 11>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__14_11() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 11>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__u_12() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 poison, i32 12>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__0_12() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 0, i32 12>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__1_12() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 1, i32 12>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__2_12() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 2, i32 12>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__3_12() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 3, i32 12>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__4_12() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 4, i32 12>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__5_12() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 5, i32 12>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__6_12() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 6, i32 12>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__7_12() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 7, i32 12>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__8_12() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 8, i32 12>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__9_12() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 12>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__10_12() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 12>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__11_12() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 12>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__12_12() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 12>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__13_12() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 12>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__14_12() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 12>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__u_13() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 poison, i32 13>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__0_13() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 0, i32 13>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__1_13() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 1, i32 13>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__2_13() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 2, i32 13>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__3_13() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 3, i32 13>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__4_13() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 4, i32 13>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__5_13() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 5, i32 13>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__6_13() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 6, i32 13>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__7_13() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 7, i32 13>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__8_13() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 8, i32 13>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__9_13() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 13>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__10_13() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 13>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__11_13() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 13>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__12_13() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 13>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__13_13() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 13>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__14_13() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 13>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__u_14() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 poison, i32 14>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__0_14() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 0, i32 14>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__1_14() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 1, i32 14>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__2_14() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 2, i32 14>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__3_14() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 3, i32 14>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__4_14() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 4, i32 14>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__5_14() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 5, i32 14>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__6_14() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 6, i32 14>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__7_14() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 7, i32 14>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__8_14() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 8, i32 14>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__9_14() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 14>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__10_14() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 14>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__11_14() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 14>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__12_14() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 14>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__13_14() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 14>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__14_14() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 14>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__u_15() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 poison, i32 15>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__0_15() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 0, i32 15>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__1_15() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 1, i32 15>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__2_15() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 2, i32 15>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__3_15() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 3, i32 15>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__4_15() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 4, i32 15>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__5_15() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 5, i32 15>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__6_15() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 6, i32 15>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__7_15() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 7, i32 15>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__8_15() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 8, i32 15>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__9_15() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 9, i32 15>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__10_15() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 10, i32 15>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__11_15() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 11, i32 15>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__12_15() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 12, i32 15>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__13_15() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 13, i32 15>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}

define void @s_shuffle_v2p3_v8p3__14_15() {
  %vec0 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %vec1 = call <8 x ptr addrspace(3)> asm "; def $0", "=s"()
  %shuf = shufflevector <8 x ptr addrspace(3)> %vec0, <8 x ptr addrspace(3)> %vec1, <2 x i32> <i32 14, i32 15>
  call void asm sideeffect "; use $0", "{s[8:9]}"(<2 x ptr addrspace(3)> %shuf)
  ret void
}
