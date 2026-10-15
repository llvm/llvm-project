; RUN: opt -S -passes='gvn,dse' < %s | FileCheck %s --check-prefix=IR
; RUN: opt -passes='gvn,dse' < %s | llc -mtriple=nvptx64 -mcpu=sm_60 -verify-machineinstrs | FileCheck %s --check-prefix=PTX --implicit-check-not=membar --implicit-check-not=fence

; Counter reads must prevent load forwarding and dead store elimination across
; the measured memory accesses, and preserve their order through code generation.

define i32 @loads(ptr addrspace(1) %p) {
; IR-LABEL: define i32 @loads(
; IR: %a = load i32, ptr addrspace(1) %p
; IR: call i32 @llvm.nvvm.read.ptx.sreg.clock()
; IR: %b = load i32, ptr addrspace(1) %p
; IR: call i64 @llvm.nvvm.read.ptx.sreg.clock64()
; IR: %c = load i32, ptr addrspace(1) %p
; IR: call i64 @llvm.nvvm.read.ptx.sreg.globaltimer()
; IR: %d = load i32, ptr addrspace(1) %p
; IR: call i32 @llvm.nvvm.read.ptx.sreg.globaltimer.lo()
; IR: %e = load i32, ptr addrspace(1) %p
; IR: call i32 @llvm.nvvm.read.ptx.sreg.pm0()
; IR: %f = load i32, ptr addrspace(1) %p
; PTX-LABEL: loads(
; PTX: ld.global.b32
; PTX: mov.u32 {{%r[0-9]+}}, %clock;
; PTX: ld.global.b32
; PTX: mov.u64 {{%rd[0-9]+}}, %clock64;
; PTX: ld.global.b32
; PTX: mov.u64 {{%rd[0-9]+}}, %globaltimer;
; PTX: ld.global.b32
; PTX: mov.u32 {{%r[0-9]+}}, %globaltimer_lo;
; PTX: ld.global.b32
; PTX: mov.u32 {{%r[0-9]+}}, %pm0;
; PTX: ld.global.b32
; PTX: ret;
  %a = load i32, ptr addrspace(1) %p
  call i32 @llvm.nvvm.read.ptx.sreg.clock()
  %b = load i32, ptr addrspace(1) %p
  call i64 @llvm.nvvm.read.ptx.sreg.clock64()
  %c = load i32, ptr addrspace(1) %p
  call i64 @llvm.nvvm.read.ptx.sreg.globaltimer()
  %d = load i32, ptr addrspace(1) %p
  call i32 @llvm.nvvm.read.ptx.sreg.globaltimer.lo()
  %e = load i32, ptr addrspace(1) %p
  call i32 @llvm.nvvm.read.ptx.sreg.pm0()
  %f = load i32, ptr addrspace(1) %p
  %ab = add i32 %a, %b
  %abc = add i32 %ab, %c
  %abcd = add i32 %abc, %d
  %abcde = add i32 %abcd, %e
  %result = add i32 %abcde, %f
  ret i32 %result
}

define void @stores(ptr addrspace(1) %p) {
; IR-LABEL: define void @stores(
; IR: store i32 0, ptr addrspace(1) %p
; IR: call i32 @llvm.nvvm.read.ptx.sreg.clock()
; IR: store i32 1, ptr addrspace(1) %p
; IR: call i64 @llvm.nvvm.read.ptx.sreg.clock64()
; IR: store i32 2, ptr addrspace(1) %p
; IR: call i64 @llvm.nvvm.read.ptx.sreg.globaltimer()
; IR: store i32 3, ptr addrspace(1) %p
; IR: call i32 @llvm.nvvm.read.ptx.sreg.globaltimer.lo()
; IR: store i32 4, ptr addrspace(1) %p
; IR: call i32 @llvm.nvvm.read.ptx.sreg.pm0()
; IR: store i32 5, ptr addrspace(1) %p
; PTX-LABEL: stores(
; PTX: st.global.b32
; PTX: mov.u32 {{%r[0-9]+}}, %clock;
; PTX: st.global.b32
; PTX: mov.u64 {{%rd[0-9]+}}, %clock64;
; PTX: st.global.b32
; PTX: mov.u64 {{%rd[0-9]+}}, %globaltimer;
; PTX: st.global.b32
; PTX: mov.u32 {{%r[0-9]+}}, %globaltimer_lo;
; PTX: st.global.b32
; PTX: mov.u32 {{%r[0-9]+}}, %pm0;
; PTX: st.global.b32
; PTX: ret;
  store i32 0, ptr addrspace(1) %p
  call i32 @llvm.nvvm.read.ptx.sreg.clock()
  store i32 1, ptr addrspace(1) %p
  call i64 @llvm.nvvm.read.ptx.sreg.clock64()
  store i32 2, ptr addrspace(1) %p
  call i64 @llvm.nvvm.read.ptx.sreg.globaltimer()
  store i32 3, ptr addrspace(1) %p
  call i32 @llvm.nvvm.read.ptx.sreg.globaltimer.lo()
  store i32 4, ptr addrspace(1) %p
  call i32 @llvm.nvvm.read.ptx.sreg.pm0()
  store i32 5, ptr addrspace(1) %p
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.clock()
declare i64 @llvm.nvvm.read.ptx.sreg.clock64()
declare i64 @llvm.nvvm.read.ptx.sreg.globaltimer()
declare i32 @llvm.nvvm.read.ptx.sreg.globaltimer.lo()
declare i32 @llvm.nvvm.read.ptx.sreg.pm0()
