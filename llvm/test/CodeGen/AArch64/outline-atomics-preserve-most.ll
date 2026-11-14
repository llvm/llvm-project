; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+outline-atomics -verify-machineinstrs < %s | FileCheck %s --check-prefix=DEFAULT
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+outline-atomics,+outline-atomics-preserve-most -verify-machineinstrs < %s | FileCheck %s --check-prefix=PRESERVE
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+outline-atomics --aarch64-outline-atomics-preserve-most -verify-machineinstrs < %s | FileCheck %s --check-prefix=PRESERVE
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+outline-atomics,+outline-atomics-preserve-most -global-isel -global-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s --check-prefix=GISEL
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -O0 -mattr=+outline-atomics,+outline-atomics-preserve-most -verify-machineinstrs < %s | FileCheck %s --check-prefix=O0
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -O0 -global-isel=0 -mattr=+outline-atomics,+outline-atomics-preserve-most -verify-machineinstrs < %s | FileCheck %s --check-prefix=O0
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+outline-atomics,+outline-atomics-preserve-most -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=CSRMASK
; RUN: llc -mtriple=aarch64-unknown-fuchsia -mattr=+outline-atomics -verify-machineinstrs < %s | FileCheck %s --check-prefix=DEFAULT
; RUN: llc -mtriple=aarch64-unknown-fuchsia -mattr=+outline-atomics,+outline-atomics-preserve-most -verify-machineinstrs < %s | FileCheck %s --check-prefix=PRESERVE
; RUN: llc -mtriple=aarch64-unknown-fuchsia -mattr=+outline-atomics --aarch64-outline-atomics-preserve-most -verify-machineinstrs < %s | FileCheck %s --check-prefix=PRESERVE
; RUN: llc -mtriple=aarch64-unknown-fuchsia -mattr=+outline-atomics,+outline-atomics-preserve-most -global-isel -global-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s --check-prefix=GISEL
; RUN: llc -mtriple=aarch64-unknown-fuchsia -O0 -mattr=+outline-atomics,+outline-atomics-preserve-most -verify-machineinstrs < %s | FileCheck %s --check-prefix=O0
; RUN: llc -mtriple=aarch64-unknown-fuchsia -O0 -global-isel=0 -mattr=+outline-atomics,+outline-atomics-preserve-most -verify-machineinstrs < %s | FileCheck %s --check-prefix=O0
; RUN: llc -mtriple=aarch64-unknown-fuchsia -mattr=+outline-atomics,+outline-atomics-preserve-most -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=CSRMASK
; RUN: llc -mtriple=aarch64-apple-darwin -mattr=+outline-atomics,+outline-atomics-preserve-most -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=CSRMASK
; RUN: llc -mtriple=aarch64-apple-darwin -mattr=+outline-atomics,+outline-atomics-preserve-most -verify-machineinstrs < %s | FileCheck %s --check-prefix=DARWIN-ASM
; RUN: llc -mtriple=arm64ec-pc-windows-msvc -mattr=+outline-atomics,+outline-atomics-preserve-most -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=ARM64EC
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+lse,+outline-atomics,+outline-atomics-preserve-most -verify-machineinstrs < %s | FileCheck %s --check-prefix=LSE
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=-outline-atomics,+outline-atomics-preserve-most -verify-machineinstrs < %s | FileCheck %s --check-prefix=NO-OUTLINE
; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+outline-atomics,+outline-atomics-preserve-most -verify-machineinstrs < %s | FileCheck %s --check-prefix=NOX5X14

; Verify that live GPRs are held in preserved x5-x14 across 1/2/4/8-byte
; outline-atomic libcalls without callee-saved stack spills.
define i64 @ldadd4_preserves_gprs(ptr %p, i32 %v, i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f) nounwind {
; DEFAULT-LABEL: ldadd4_preserves_gprs:
; DEFAULT:       stp x24, x23,
; DEFAULT:       stp x22, x21,
; DEFAULT:       stp x20, x19,
; DEFAULT:       bl __aarch64_ldadd4_acq_rel{{$}}
;
; PRESERVE-LABEL: ldadd4_preserves_gprs:
; PRESERVE-NOT:   stp x
; PRESERVE:       bl __aarch64_ldadd4_acq_rel_preservemost{{$}}
; PRESERVE-NOT:   ldp x
; PRESERVE:       ret
;
; DARWIN-ASM-LABEL: _ldadd4_preserves_gprs:
; DARWIN-ASM-NOT:   x19
; DARWIN-ASM-NOT:   x20
; DARWIN-ASM:       bl ___aarch64_ldadd4_acq_rel_preservemost{{$}}
; DARWIN-ASM:       ret
;
; LSE-LABEL: ldadd4_preserves_gprs:
; LSE-NOT:   __aarch64_
; LSE:       ldaddal w{{[0-9]+}}, w{{[0-9]+}}, [x0]
;
; NO-OUTLINE-LABEL: ldadd4_preserves_gprs:
; NO-OUTLINE-NOT:   __aarch64_
; NO-OUTLINE:       ldaxr w{{[0-9]+}}, [x0]
; NO-OUTLINE:       stlxr w{{[0-9]+}}, w{{[0-9]+}}, [x0]
;
; GISEL-LABEL: ldadd4_preserves_gprs:
; GISEL-NOT:   stp x
; GISEL:       bl __aarch64_ldadd4_acq_rel_preservemost
; GISEL:       ret
;
; CSRMASK-LABEL: name: ldadd4_preserves_gprs
; CSRMASK: BL &__aarch64_ldadd4_acq_rel_preservemost, csr_aarch64_outline_atomics_preservemost
; ARM64EC-LABEL: name: '#ldadd4_preserves_gprs'
; ARM64EC: BL &"#__aarch64_ldadd4_acq_rel", csr_aarch64_aapcs
  %old = atomicrmw add ptr %p, i32 %v acq_rel
  %after = load volatile i32, ptr %p
  %sum0 = add i32 %old, %after
  %sum0z = zext i32 %sum0 to i64
  %s1 = add i64 %a, %b
  %s2 = add i64 %c, %d
  %s3 = add i64 %e, %f
  %s4 = add i64 %s1, %s2
  %s5 = add i64 %s3, %sum0z
  %res = add i64 %s4, %s5
  ret i64 %res
}

; Verify that 16-byte CAS uses the same calling convention, keeping live values
; in x5-x14 without callee-saved stack spills while treating x0-x4 as clobbered.
define i128 @cas16_preserves_gprs(ptr %p, i128 %exp, i128 %des, i64 %extra) nounwind {
; DEFAULT-LABEL: cas16_preserves_gprs:
; DEFAULT:       stp {{.*}}x19
; DEFAULT:       bl __aarch64_cas16_acq_rel{{$}}
;
; PRESERVE-LABEL: cas16_preserves_gprs:
; PRESERVE-NOT:   x19
; PRESERVE-NOT:   x20
; PRESERVE:       bl __aarch64_cas16_acq_rel_preservemost
; PRESERVE:       ret
;
; GISEL-LABEL: cas16_preserves_gprs:
; GISEL-NOT:   x19
; GISEL-NOT:   x20
; GISEL:       bl __aarch64_cas16_acq_rel_preservemost
; GISEL:       ret
;
; CSRMASK-LABEL: name: cas16_preserves_gprs
; CSRMASK: BL &__aarch64_cas16_acq_rel_preservemost, csr_aarch64_outline_atomics_preservemost
; ARM64EC-LABEL: name: '#cas16_preserves_gprs'
; ARM64EC: BL &"#__aarch64_cas16_acq_rel", csr_aarch64_aapcs
  %pair = cmpxchg ptr %p, i128 %exp, i128 %des acq_rel acquire
  %old = extractvalue { i128, i1 } %pair, 0
  %ext = zext i64 %extra to i128
  %sum1 = add i128 %old, %des
  %sum2 = add i128 %sum1, %ext
  store volatile i128 %sum2, ptr %p
  ret i128 %sum2
}

; Verify that NEON 128-bit vector registers (q0-q31) are preserved without spills.
define <4 x i32> @swp8_preserves_simd(ptr %p, i64 %v, <4 x i32> %vec0, <4 x i32> %vec1) nounwind {
; DEFAULT-LABEL: swp8_preserves_simd:
; DEFAULT:       stp q{{[0-9]+}}, q{{[0-9]+}}, [sp
; DEFAULT:       bl __aarch64_swp8_rel{{$}}
;
; PRESERVE-LABEL: swp8_preserves_simd:
; PRESERVE-NOT:   stp q
; PRESERVE-NOT:   str q
; PRESERVE:       bl __aarch64_swp8_rel_preservemost
; PRESERVE-NOT:   ldp q
; PRESERVE-NOT:   ldr q
; PRESERVE:       add v0.4s, v0.4s, v1.4s
; PRESERVE:       ret
  %old = atomicrmw xchg ptr %p, i64 %v release
  %res = add <4 x i32> %vec0, %vec1
  ret <4 x i32> %res
}

; Verify the exact register boundaries:
; - x4 (argument register) and x15-x17 (scratch) and NZCV are clobbered;
; - x5 and x14 are preserved in place across the call.
define i64 @exact_x4_x5_x14_x15_x16_x17_and_nzcv_boundary(ptr %p, i32 %v, i64 %x, i64 %y) nounwind {
; PRESERVE-LABEL: exact_x4_x5_x14_x15_x16_x17_and_nzcv_boundary:
; PRESERVE-NOT:   mov {{x[0-9]+}}, x5
; PRESERVE-NOT:   mov {{x[0-9]+}}, x14
; PRESERVE-DAG:   mov [[SAVEX4:x[6-9]|x1[0-3]]], x4
; PRESERVE-DAG:   mov [[SAVEX15:x[6-9]|x1[0-3]]], x15
; PRESERVE-DAG:   mov [[SAVEX16:x[6-9]|x1[0-3]]], x16
; PRESERVE-DAG:   mov [[SAVEX17:x[6-9]|x1[0-3]]], x17
; PRESERVE:       bl __aarch64_ldclr4_relax_preservemost
; PRESERVE-DAG:   mov x4, [[SAVEX4]]
; PRESERVE-DAG:   mov x15, [[SAVEX15]]
; PRESERVE-DAG:   mov x16, [[SAVEX16]]
; PRESERVE-DAG:   mov x17, [[SAVEX17]]
; PRESERVE-DAG:   cmp {{x[0-9]+}}, {{x[0-9]+}}
; PRESERVE-DAG:   csel x0,
; PRESERVE-NOT:   mov x5,
; PRESERVE-NOT:   mov x14,
; PRESERVE:       ret
;
; O0-LABEL: exact_x4_x5_x14_x15_x16_x17_and_nzcv_boundary:
; O0:       bl __aarch64_ldclr4_relax_preservemost
;
; NOX5X14-LABEL: exact_x4_x5_x14_x15_x16_x17_and_nzcv_boundary:
; NOX5X14-NOT:   {{mov|str|stp|ldr|ldp}} {{.*}}x5{{[,\]\n]|$}}
; NOX5X14-NOT:   {{mov|str|stp|ldr|ldp}} {{.*}}x14{{[,\]\n]|$}}
; NOX5X14:       ret
  %cmp = icmp eq i64 %x, %y
  %r4 = call i64 asm sideeffect "", "={x4}"()
  %r5 = call i64 asm sideeffect "", "={x5}"()
  %r14 = call i64 asm sideeffect "", "={x14}"()
  %r15 = call i64 asm sideeffect "", "={x15}"()
  %r16 = call i64 asm sideeffect "", "={x16}"()
  %r17 = call i64 asm sideeffect "", "={x17}"()
  %old = atomicrmw and ptr %p, i32 %v monotonic
  call void asm sideeffect "", "{x4},{x5},{x14},{x15},{x16},{x17}"(i64 %r4, i64 %r5, i64 %r14, i64 %r15, i64 %r16, i64 %r17)
  %old64 = zext i32 %old to i64
  %sel = select i1 %cmp, i64 %x, i64 %old64
  ret i64 %sel
}
