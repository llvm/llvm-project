@ RUN: not llvm-mc < %s -triple thumbv8-unknown-unknown -show-encoding -mattr=+fp-only-sp,-neon 2> %t > %t2
@ RUN:     FileCheck %s < %t --check-prefix=CHECK-ERRORS
@ RUN:     FileCheck %s < %t2

        vadd.f64 d0, d1, d2
        vsub.f64 d2, d3, d4
        vdiv.f64 d4, d5, d6
        vmul.f64 d6, d7, d8
        vnmul.f64 d8, d9, d10
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vadd.f64 d0, d1, d2
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vsub.f64 d2, d3, d4
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vdiv.f64 d4, d5, d6
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vmul.f64 d6, d7, d8
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vnmul.f64 d8, d9, d10

        vmla.f64 d11, d10, d9
        vmls.f64 d8, d7, d6
        vnmla.f64 d5, d4, d3
        vnmls.f64 d2, d1, d0
        vfma.f64 d1, d2, d3
        vfms.f64 d4, d5, d6
        vfnma.f64 d7, d8, d9
        vfnms.f64 d10, d11, d12
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vmla.f64 d11, d10, d9
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vmls.f64 d8, d7, d6
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vnmla.f64 d5, d4, d3
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vnmls.f64 d2, d1, d0
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vfma.f64 d1, d2, d3
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vfms.f64 d4, d5, d6
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vfnma.f64 d7, d8, d9
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vfnms.f64 d10, d11, d12

        vneg.f64 d15, d14
        vsqrt.f64 d13, d12
        vsqrt d13, d14
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vneg.f64 d15, d14
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vsqrt.f64 d13, d12
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vsqrt d13, d14

        vcmpe.f64 d0, d1
        vcmp.f64 d2, d3
        vabs.f64 d4, d5
        vcmpe.f64 d5, #0
        vcmp.f64 d6, #0
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcmpe.f64 d0, d1
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcmp.f64 d2, d3
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vabs.f64 d4, d5
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcmpe.f64 d5, #0
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcmp.f64 d6, #0

        @ FIXME: overlapping aliases and a probable TableGen indeterminacy mean
        @ that the actual reason can vary by platform.
        vmov.f64 d11, d10
@ CHECK-ERRORS: instruction requires: NEON
@ CHECK-ERRORS-NEXT: vmov.f64 d11, d10

        vcvt.f64.s32 d9, s8
        vcvt.f64.u32 d7, s6
        vcvt.s32.f64 s5, d4
        vcvt.u32.f64 s3, d2
        vcvtr.s32.f64 s1, d0
        vcvtr.u32.f64 s1, d2
        vcvt.s16.f64 d3, d4, #1
        vcvt.u16.f64 d5, d6, #2
        vcvt.s32.f64 d7, d8, #3
        vcvt.u32.f64 d9, d10, #4
        vcvt.f64.s16 d11, d12, #3
        vcvt.f64.u16 d13, d14, #2
        vcvt.f64.s32 d15, d14, #1
        vcvt.f64.u32 d13, d12, #1
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcvt.f64.s32 d9, s8
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcvt.f64.u32 d7, s6
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcvt.s32.f64 s5, d4
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcvt.u32.f64 s3, d2
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcvtr.s32.f64 s1, d0
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcvtr.u32.f64 s1, d2
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcvt.s16.f64 d3, d4, #1
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcvt.u16.f64 d5, d6, #2
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcvt.s32.f64 d7, d8, #3
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcvt.u32.f64 d9, d10, #4
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcvt.f64.s16 d11, d12, #3
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcvt.f64.u16 d13, d14, #2
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcvt.f64.s32 d15, d14, #1
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcvt.f64.u32 d13, d12, #1

        @ v8 operations, also double precision so make sure they're rejected.
        vselgt.f64 d0, d1, d2
        vselge.f64 d3, d4, d5
        vseleq.f64 d6, d7, d8
        vselvs.f64 d9, d10, d11
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vselgt.f64 d0, d1, d2
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vselge.f64 d3, d4, d5
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vseleq.f64 d6, d7, d8
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vselvs.f64 d9, d10, d11

        vmaxnm.f64 d12, d13, d14
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vmaxnm.f64 d12, d13, d14

        vcvtb.f64.f16 d7, s8
        vcvtb.f16.f64 s9, d10
        vcvtt.f64.f16 d11, s12
        vcvtt.f16.f64 s13, d14
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcvtb.f64.f16 d7, s8
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcvtb.f16.f64 s9, d10
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcvtt.f64.f16 d11, s12
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vcvtt.f16.f64 s13, d14

        vrintz.f64 d15, d14
        vrintr.f64.f64 d13, d12
        vrintx.f64 d11, d10
        vrinta.f64.f64 d9, d8
        vrintn.f64 d7, d6
        vrintp.f64.f64 d5, d4
        vrintm.f64 d3, d2
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vrintz.f64 d15, d14
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vrintr.f64.f64 d13, d12
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vrintx.f64 d11, d10
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vrinta.f64.f64 d9, d8
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vrintn.f64 d7, d6
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vrintp.f64.f64 d5, d4
@ CHECK-ERRORS: error: instruction requires: double precision VFP
@ CHECK-ERRORS-NEXT: vrintm.f64 d3, d2

        @ Double precisionish operations that actually *are* allowed.
        vldr d0, [sp]
        vstr d3, [sp]
        vldm r0, {d0, d1}
        vstm r4, {d3, d4}
        vpush {d6, d7}
        vpop {d8, d9}
        vmov r1, r0, d1
        vmov d2, r3, r4
        vmov.f64 r5, r6, d7
        vmov.f64 d8, r9, r10
@ CHECK: vldr d0, [sp]
@ CHECK: vstr d3, [sp]
@ CHECK: vldmia r0, {d0, d1}
@ CHECK: vstmia r4, {d3, d4}
@ CHECK: vpush {d6, d7}
@ CHECK: vpop {d8, d9}
@ CHECK: vmov r1, r0, d1
@ CHECK: vmov d2, r3, r4
@ CHECK: vmov r5, r6, d7
@ CHECK: vmov d8, r9, r10
