; Check that custom inserters and frame lowering mark the CC defs of
; instructions whose condition code is never read as dead.

; RUN: llc -mtriple=s390x-linux-gnu -stop-after=finalize-isel -o - %s | FileCheck %s
; RUN: llc -mtriple=s390x-linux-gnu -stop-after=prolog-epilog -o - %s | FileCheck -check-prefix=PEI %s

; CHECK-LABEL: name: atomicrmw_and_i8
; CHECK: NR {{.*}}, implicit-def dead $cc
define i8 @atomicrmw_and_i8(ptr %src, i8 %b) {
  %res = atomicrmw and ptr %src, i8 %b seq_cst
  ret i8 %res
}

; CHECK-LABEL: name: atomicrmw_nand_i8
; CHECK: NR {{.*}}, implicit-def dead $cc
; CHECK: XILF {{.*}}, implicit-def dead $cc
define i8 @atomicrmw_nand_i8(ptr %src, i8 %b) {
  %res = atomicrmw nand ptr %src, i8 %b seq_cst
  ret i8 %res
}

; CHECK-LABEL: name: atomicrmw_xchg_i8
; CHECK: RISBG32 {{.*}}, implicit-def dead $cc
define i8 @atomicrmw_xchg_i8(ptr %src, i8 %b) {
  %res = atomicrmw xchg ptr %src, i8 %b seq_cst
  ret i8 %res
}

; CHECK-LABEL: name: atomicrmw_max_i8
; CHECK: RISBG32 {{.*}}, implicit-def dead $cc
define i8 @atomicrmw_max_i8(ptr %src, i8 %b) {
  %res = atomicrmw max ptr %src, i8 %b seq_cst
  ret i8 %res
}

; CHECK-LABEL: name: cmpxchg_i8
; CHECK: RISBG32 {{.*}}, implicit-def dead $cc
define i8 @cmpxchg_i8(ptr %src, i8 %cmp, i8 %swap) {
  %pair = cmpxchg ptr %src, i8 %cmp, i8 %swap seq_cst seq_cst
  %res = extractvalue { i8, i1 } %pair, 0
  ret i8 %res
}

; CHECK-LABEL: name: memcpy_variable
; CHECK: AGHI {{.*}}, -1, implicit-def dead $cc
; CHECK: CGHI {{.*}}, 0, implicit-def $cc
define void @memcpy_variable(ptr %dest, ptr %src, i64 %len) {
  call void @llvm.memcpy.p0.p0.i64(ptr %dest, ptr %src, i64 %len, i1 false)
  ret void
}

; CHECK-LABEL: name: probed_dynamic_alloca
; CHECK: CLGFI {{.*}}, 4096, implicit-def $cc
; CHECK: SLGFI {{.*}}, 4096, implicit-def dead $cc
; CHECK: $r15d = SLGFI $r15d, 4096, implicit-def dead $cc
; CHECK: CG $r15d, $r15d, 4088, $noreg, implicit-def dead $cc
; CHECK: CGHI {{.*}}, 0, implicit-def $cc
; CHECK: $r15d = SLGR $r15d, {{.*}}, implicit-def dead $cc
; CHECK: CG $r15d, $r15d, -8, {{.*}}, implicit-def dead $cc
define ptr @probed_dynamic_alloca(i64 %len) "probe-stack"="inline-asm" {
  %p = alloca i8, i64 %len
  ret ptr %p
}

; PEI-LABEL: name: probed_static_alloca
; PEI: $r15d = AGHI $r15d, -4096, implicit-def dead $cc
; PEI: CG undef $r0d, $r15d, 4088, $noreg, implicit-def dead $cc
define void @probed_static_alloca() "probe-stack"="inline-asm" {
  %p = alloca [4000 x i8]
  call void @use(ptr %p)
  ret void
}

declare void @use(ptr)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
