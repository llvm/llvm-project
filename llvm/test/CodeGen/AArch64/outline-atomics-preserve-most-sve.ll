; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+sve,+outline-atomics -verify-machineinstrs < %s | FileCheck %s --check-prefix=DEFAULT
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+sve,+outline-atomics,+outline-atomics-preserve-most -verify-machineinstrs < %s | FileCheck %s --check-prefix=PRESERVE
; RUN: llc -mtriple=aarch64-unknown-fuchsia -mattr=+sve,+outline-atomics -verify-machineinstrs < %s | FileCheck %s --check-prefix=DEFAULT
; RUN: llc -mtriple=aarch64-unknown-fuchsia -mattr=+sve,+outline-atomics,+outline-atomics-preserve-most -verify-machineinstrs < %s | FileCheck %s --check-prefix=PRESERVE

; Verify that SVE scalable vector (z0-z31) and predicate (p0-p15) registers are
; preserved across outline-atomic libcalls without scalable stack spills.
define <vscale x 4 x i32> @ldset8_preserves_sve(ptr %p, i64 %v, <vscale x 4 x i32> %sv0, <vscale x 4 x i32> %sv1, <vscale x 4 x i1> %pg) nounwind {
; DEFAULT-LABEL: ldset8_preserves_sve:
; DEFAULT:       addvl sp, sp,
; DEFAULT:       str z{{[0-9]+}},
; DEFAULT:       bl __aarch64_ldset8_acq_rel{{$}}
;
; PRESERVE-LABEL: ldset8_preserves_sve:
; PRESERVE-NOT:   addvl
; PRESERVE-NOT:   str z
; PRESERVE-NOT:   str p
; PRESERVE:       bl __aarch64_ldset8_acq_rel_preservemost{{$}}
; PRESERVE-NOT:   ldr z
; PRESERVE-NOT:   ldr p
; PRESERVE:       add z0.s, p0/m, z0.s, z1.s
; PRESERVE:       ret
  %old = atomicrmw or ptr %p, i64 %v acq_rel
  %res = call <vscale x 4 x i32> @llvm.aarch64.sve.add.nxv4i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %sv0, <vscale x 4 x i32> %sv1)
  ret <vscale x 4 x i32> %res
}

declare <vscale x 4 x i32> @llvm.aarch64.sve.add.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x i32>, <vscale x 4 x i32>)
