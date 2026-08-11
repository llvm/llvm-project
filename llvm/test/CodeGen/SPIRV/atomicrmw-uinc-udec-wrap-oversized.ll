; The SPIR-V target supports no atomic wider than 64 bits
; (SPIRVTargetLowering sets setMaxAtomicSizeInBitsSupported(64)), so a vector
; operand that does not fit is rejected by AtomicExpandPass before the
; uinc_wrap/udec_wrap helper lowering ever sees it. Verify that carrying these
; two operations across as an imported helper on AMD targets does not smuggle
; an over-limit atomic past that check: the diagnostic is the same one any
; other target and any other atomicrmw operation gets. The supported vector
; widths are covered by atomicrmw-uinc-udec-wrap-signatures.ll.

; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - 2>&1 | FileCheck %s
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - 2>&1 | FileCheck %s

; CHECK: error: unsupported atomicrmw uinc_wrap: target supports atomics up to 8 bytes, but this atomic accesses 16 bytes
; CHECK-NOT: __translate_spirv_atomic

@uv = common dso_local addrspace(1) global <4 x i32> zeroinitializer, align 16

define dso_local spir_func void @oversized_vector() local_unnamed_addr {
entry:
  %v = atomicrmw uinc_wrap ptr addrspace(1) @uv, <4 x i32> splat (i32 42) monotonic
  ret void
}
