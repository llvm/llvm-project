; agx-lower-air drops PTX's explicit .ftz flushes written on the bits (a
; front end's strict mode tests the bits: a flushing GPU compares a denormal
; as zero) where the f32 ALU flushes anyway: the flush of an fma's result,
; also through its canonical NaN select, and the flush of an operand that only
; an fma reads. A flush of a value that no flushing operation makes or reads
; stays.
; RUN: llc -mtriple=air -air-portable %s -o %t.air.ll
; RUN: opt -mtriple=agx -mcpu=agx-g14g -passes=agx-lower-air -S %t.air.ll | FileCheck %s

; CHECK-LABEL: define agx_kernel void @k(
; CHECK-NOT: select i1 %zt
; CHECK: %r = call float @llvm.fma.f32(float %x.val, float %y.val, float %zff)
; CHECK: %rn = icmp ugt i32 %ra, 2139095040
; CHECK-NEXT: %rf = select i1 %rn, i32 2147483647, i32 %rb
; CHECK-NEXT: store i32 %rf
; CHECK: %wf = select i1 %wt, i32 %ws, i32 %w.val
; CHECK: ret void
define void @k(ptr addrspace(1) %out, float %x, float %y, float %z, i32 %w) "air-kernel" "denormal-fp-math-f32"="preservesign,preservesign" {
  ; the operand flushed, read only by the fma
  %zb = bitcast float %z to i32
  %ze = and i32 %zb, 2139095040
  %zt = icmp eq i32 %ze, 0
  %zs = and i32 %zb, -2147483648
  %zf = select i1 %zt, i32 %zs, i32 %zb
  %zff = bitcast i32 %zf to float
  %r = call float @llvm.fma.f32(float %x, float %y, float %zff)
  ; the result canonicalized and flushed
  %rb = bitcast float %r to i32
  %ra = and i32 %rb, 2147483647
  %rn = icmp ugt i32 %ra, 2139095040
  %rc = select i1 %rn, i32 2147483647, i32 %rb
  %re = and i32 %rc, 2139095040
  %rt = icmp eq i32 %re, 0
  %rs = and i32 %rc, -2147483648
  %rf = select i1 %rt, i32 %rs, i32 %rc
  store i32 %rf, ptr addrspace(1) %out
  ; an integer flushed as if it were a float: stays
  %we = and i32 %w, 2139095040
  %wt = icmp eq i32 %we, 0
  %ws = and i32 %w, -2147483648
  %wf = select i1 %wt, i32 %ws, i32 %w
  %o1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  store i32 %wf, ptr addrspace(1) %o1
  ret void
}

declare float @llvm.fma.f32(float, float, float)

