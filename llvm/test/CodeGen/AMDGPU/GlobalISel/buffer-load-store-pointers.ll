; RUN: not --crash llc -global-isel -mtriple=amdgcn -mcpu=gfx90a -verify-machineinstrs -stop-after=instruction-select < %s

define ptr addrspace(0) @buffer_load_p0(ptr addrspace(8) inreg %buf) {
  %ret = call ptr addrspace(0) @llvm.amdgcn.raw.ptr.buffer.load.p0(ptr addrspace(8) inreg %buf, i32 0, i32 0, i32 0)
  ret ptr addrspace(0) %ret
}

define void @buffer_store_p0(ptr addrspace(0) %data, ptr addrspace(8) inreg %buf) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.p0(ptr addrspace(0) %data, ptr addrspace(8) inreg %buf, i32 0, i32 0, i32 0)
  ret void
}

define ptr addrspace(1) @buffer_load_p1(ptr addrspace(8) inreg %buf) {
  %ret = call ptr addrspace(1) @llvm.amdgcn.raw.ptr.buffer.load.p1(ptr addrspace(8) inreg %buf, i32 0, i32 0, i32 0)
  ret ptr addrspace(1) %ret
}

define void @buffer_store_p1(ptr addrspace(1) %data, ptr addrspace(8) inreg %buf) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.p1(ptr addrspace(1) %data, ptr addrspace(8) inreg %buf, i32 0, i32 0, i32 0)
  ret void
}

define ptr addrspace(4) @buffer_load_p4(ptr addrspace(8) inreg %buf) {
  %ret = call ptr addrspace(4) @llvm.amdgcn.raw.ptr.buffer.load.p4(ptr addrspace(8) inreg %buf, i32 0, i32 0, i32 0)
  ret ptr addrspace(4) %ret
}

define void @buffer_store_p4(ptr addrspace(4) %data, ptr addrspace(8) inreg %buf) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.p4(ptr addrspace(4) %data, ptr addrspace(8) inreg %buf, i32 0, i32 0, i32 0)
  ret void
}

define ptr addrspace(5) @buffer_load_p5(ptr addrspace(8) inreg %buf) {
  %ret = call ptr addrspace(5) @llvm.amdgcn.raw.ptr.buffer.load.p5(ptr addrspace(8) inreg %buf, i32 0, i32 0, i32 0)
  ret ptr addrspace(5) %ret
}

define void @buffer_store_p5(ptr addrspace(5) %data, ptr addrspace(8) inreg %buf) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.p5(ptr addrspace(5) %data, ptr addrspace(8) inreg %buf, i32 0, i32 0, i32 0)
  ret void
}

define <2 x ptr addrspace(1)> @buffer_load_v2p1(ptr addrspace(8) inreg %buf) {
  %ret = call <2 x ptr addrspace(1)> @llvm.amdgcn.raw.ptr.buffer.load.v2p1(ptr addrspace(8) inreg %buf, i32 0, i32 0, i32 0)
  ret <2 x ptr addrspace(1)> %ret
}

define void @buffer_store_v2p5(<2 x ptr addrspace(1)> %data, ptr addrspace(8) inreg %buf) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.v2p1(<2 x ptr addrspace(1)> %data, ptr addrspace(8) inreg %buf, i32 0, i32 0, i32 0)
  ret void
}

define <3 x ptr addrspace(5)> @buffer_load_v3p5(ptr addrspace(8) inreg %buf) {
  %ret = call <3 x ptr addrspace(5)> @llvm.amdgcn.raw.ptr.buffer.load.v3p5(ptr addrspace(8) inreg %buf, i32 0, i32 0, i32 0)
  ret <3 x ptr addrspace(5)> %ret
}

define void @buffer_store_v3p5(<3 x ptr addrspace(5)> %data, ptr addrspace(8) inreg %buf) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.v3p5(<3 x ptr addrspace(5)> %data, ptr addrspace(8) inreg %buf, i32 0, i32 0, i32 0)
  ret void
}

define <4 x ptr addrspace(5)> @buffer_load_v4p5(ptr addrspace(8) inreg %buf) {
  %ret = call <4 x ptr addrspace(5)> @llvm.amdgcn.raw.ptr.buffer.load.v4p5(ptr addrspace(8) inreg %buf, i32 0, i32 0, i32 0)
  ret <4 x ptr addrspace(5)> %ret
}

define void @buffer_store_v4p5(<4 x ptr addrspace(5)> %data, ptr addrspace(8) inreg %buf) {
  call void @llvm.amdgcn.raw.ptr.buffer.store.v4p5(<4 x ptr addrspace(5)> %data, ptr addrspace(8) inreg %buf, i32 0, i32 0, i32 0)
  ret void
}
