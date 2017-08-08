; This tests that MC/asm header conversion is smooth and that the
; build attributes are correct

; RUN: llc < %s -mtriple=thumbv5-linux-gnueabi -mcpu=xscale -mattr=+strict-align | FileCheck %s --check-prefix=XSCALE
; RUN: llc < %s -mtriple=armv6-linux-gnueabi -mattr=+strict-align | FileCheck %s --check-prefix=V6
; RUN: llc < %s -mtriple=armv6-linux-gnueabi -mattr=+strict-align  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=V6-FAST
; RUN: llc < %s -mtriple=armv6-linux-gnueabi -mattr=+strict-align -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=thumbv6m-linux-gnueabi -mattr=+strict-align | FileCheck %s --check-prefix=V6M
; RUN: llc < %s -mtriple=thumbv6m-linux-gnueabi -mattr=+strict-align  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=V6M-FAST
; RUN: llc < %s -mtriple=thumbv6sm-linux-gnueabi -mattr=+strict-align | FileCheck %s --check-prefix=V6M
; RUN: llc < %s -mtriple=thumbv6sm-linux-gnueabi -mattr=+strict-align -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=V6M-FAST
; RUN: llc < %s -mtriple=armv6-linux-gnueabi -mcpu=arm1156t2f-s -mattr=+strict-align | FileCheck %s --check-prefix=ARM1156T2F-S
; RUN: llc < %s -mtriple=armv6-linux-gnueabi -mcpu=arm1156t2f-s -mattr=+strict-align  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast  | FileCheck %s --check-prefix=ARM1156T2F-S-FAST
; RUN: llc < %s -mtriple=armv6-linux-gnueabi -mcpu=arm1156t2f-s -mattr=+strict-align -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi | FileCheck %s --check-prefix=V7M
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=V7M-FAST
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv7-linux-gnueabi | FileCheck %s --check-prefix=V7
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv7-linux-gnueabi  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=V7-FAST
; RUN: llc < %s -mtriple=armv8-linux-gnueabi | FileCheck %s --check-prefix=V8
; RUN: llc < %s -mtriple=armv8-linux-gnueabi  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=V8-FAST
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=thumbv8-linux-gnueabi | FileCheck %s --check-prefix=Vt8
; RUN: llc < %s -mtriple=thumbv8-linux-gnueabi -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mattr=-neon,-crypto | FileCheck %s --check-prefix=V8-FPARMv8
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mattr=-fp-armv8,-crypto | FileCheck %s --check-prefix=V8-NEON
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mattr=-crypto | FileCheck %s --check-prefix=V8-FPARMv8-NEON
; RUN: llc < %s -mtriple=armv8-linux-gnueabi | FileCheck %s --check-prefix=V8-FPARMv8-NEON-CRYPTO
; RUN: llc < %s -mtriple=thumbv8m.base-linux-gnueabi | FileCheck %s --check-prefix=V8MBASELINE
; RUN: llc < %s -mtriple=thumbv8m.main-linux-gnueabi | FileCheck %s --check-prefix=V8MMAINLINE
; RUN: llc < %s -mtriple=thumbv8m.main-linux-gnueabi -mattr=+dsp | FileCheck %s --check-prefix=V8MMAINLINE_DSP
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a5 | FileCheck %s --check-prefix=CORTEX-A5-DEFAULT
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a5  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-A5-DEFAULT-FAST
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a5 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a5 -mattr=-neon,+d16 | FileCheck %s --check-prefix=CORTEX-A5-NONEON
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a5 -mattr=-vfp2 | FileCheck %s --check-prefix=CORTEX-A5-NOFPU
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a5 -mattr=-vfp2  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-A5-NOFPU-FAST
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a8 -float-abi=soft | FileCheck %s --check-prefix=CORTEX-A8-SOFT
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a8 -float-abi=soft  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-A8-SOFT-FAST
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a8 -float-abi=hard | FileCheck %s --check-prefix=CORTEX-A8-HARD
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a8 -float-abi=hard  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-A8-HARD-FAST
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a8 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a8 -float-abi=soft | FileCheck %s --check-prefix=CORTEX-A8-SOFT
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a9 -float-abi=soft | FileCheck %s --check-prefix=CORTEX-A9-SOFT
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a9 -float-abi=soft  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-A9-SOFT-FAST
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a9 -float-abi=hard | FileCheck %s --check-prefix=CORTEX-A9-HARD
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a9 -float-abi=hard  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-A9-HARD-FAST
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a9 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a12 | FileCheck %s --check-prefix=CORTEX-A12-DEFAULT
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a9 -float-abi=soft | FileCheck %s --check-prefix=CORTEX-A9-SOFT
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a12  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-A12-DEFAULT-FAST
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a12 -mattr=-vfp2 | FileCheck %s --check-prefix=CORTEX-A12-NOFPU
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a12 -mattr=-vfp2  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-A12-NOFPU-FAST
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a12 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15 | FileCheck %s --check-prefix=CORTEX-A15
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-A15-FAST
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a17 | FileCheck %s --check-prefix=CORTEX-A17-DEFAULT
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a17  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-A17-FAST
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a17 -mattr=-vfp2 | FileCheck %s --check-prefix=CORTEX-A17-NOFPU
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a17 -mattr=-vfp2  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-A17-NOFPU-FAST

; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15 -enable-no-trapping-fp-math | FileCheck %s --check-prefix=NO-TRAPPING-MATH
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15 -denormal-fp-math=ieee | FileCheck %s --check-prefix=DENORMAL-IEEE
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15 -denormal-fp-math=preserve-sign | FileCheck %s --check-prefix=DENORMAL-PRESERVE-SIGN
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a15 -denormal-fp-math=positive-zero | FileCheck %s --check-prefix=DENORMAL-POSITIVE-ZERO

; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mattr=-neon,+vfp3,+fp16 | FileCheck %s --check-prefix=GENERIC-FPU-VFPV3-FP16
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mattr=-neon,+vfp3,+d16,+fp16 | FileCheck %s --check-prefix=GENERIC-FPU-VFPV3-D16-FP16
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mattr=-neon,+vfp3,+fp-only-sp,+d16 | FileCheck %s --check-prefix=GENERIC-FPU-VFPV3XD
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mattr=-neon,+vfp3,+fp-only-sp,+d16,+fp16 | FileCheck %s --check-prefix=GENERIC-FPU-VFPV3XD-FP16
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mattr=+neon,+fp16 | FileCheck %s --check-prefix=GENERIC-FPU-NEON-FP16

; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a17 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=thumbv6m-linux-gnueabi -mcpu=cortex-m0 -mattr=+strict-align | FileCheck %s --check-prefix=CORTEX-M0
; RUN: llc < %s -mtriple=thumbv6m-linux-gnueabi -mcpu=cortex-m0 -mattr=+strict-align  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-M0-FAST
; RUN: llc < %s -mtriple=thumbv6m-linux-gnueabi -mcpu=cortex-m0 -mattr=+strict-align -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=thumbv6m-linux-gnueabi -mcpu=cortex-m0plus -mattr=+strict-align | FileCheck %s --check-prefix=CORTEX-M0PLUS
; RUN: llc < %s -mtriple=thumbv6m-linux-gnueabi -mcpu=cortex-m0plus -mattr=+strict-align  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-M0PLUS-FAST
; RUN: llc < %s -mtriple=thumbv6m-linux-gnueabi -mcpu=cortex-m0plus -mattr=+strict-align -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=thumbv6m-linux-gnueabi -mcpu=cortex-m1 -mattr=+strict-align | FileCheck %s --check-prefix=CORTEX-M1
; RUN: llc < %s -mtriple=thumbv6m-linux-gnueabi -mcpu=cortex-m1 -mattr=+strict-align  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-M1-FAST
; RUN: llc < %s -mtriple=thumbv6m-linux-gnueabi -mcpu=cortex-m1 -mattr=+strict-align -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=thumbv6m-linux-gnueabi -mcpu=sc000 -mattr=+strict-align | FileCheck %s --check-prefix=SC000
; RUN: llc < %s -mtriple=thumbv6m-linux-gnueabi -mcpu=sc000 -mattr=+strict-align  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=SC000-FAST
; RUN: llc < %s -mtriple=thumbv6m-linux-gnueabi -mcpu=sc000 -mattr=+strict-align -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi -mcpu=cortex-m3 | FileCheck %s --check-prefix=CORTEX-M3
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi -mcpu=cortex-m3  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-M3-FAST
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi -mcpu=cortex-m3 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi -mcpu=sc300 | FileCheck %s --check-prefix=SC300
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi -mcpu=sc300  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=SC300-FAST
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi -mcpu=sc300 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi -mcpu=cortex-m4 -float-abi=soft | FileCheck %s --check-prefix=CORTEX-M4-SOFT
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi -mcpu=cortex-m4 -float-abi=soft  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-M4-SOFT-FAST
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi -mcpu=cortex-m4 -float-abi=hard | FileCheck %s --check-prefix=CORTEX-M4-HARD
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi -mcpu=cortex-m4 -float-abi=hard  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-M4-HARD-FAST
; RUN: llc < %s -mtriple=thumbv7m-linux-gnueabi -mcpu=cortex-m4 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=thumbv7em-linux-gnueabi -mcpu=cortex-m7 -mattr=-vfp2 | FileCheck %s --check-prefix=CORTEX-M7 --check-prefix=CORTEX-M7-SOFT
; RUN: llc < %s -mtriple=thumbv7em-linux-gnueabi -mcpu=cortex-m7 -mattr=-vfp2  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-M7-NOFPU-FAST
; RUN: llc < %s -mtriple=thumbv7em-linux-gnueabi -mcpu=cortex-m7 -mattr=+fp-only-sp | FileCheck %s --check-prefix=CORTEX-M7 --check-prefix=CORTEX-M7-SINGLE
; RUN: llc < %s -mtriple=thumbv7em-linux-gnueabi -mcpu=cortex-m7 -mattr=+fp-only-sp  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-M7-FAST
; RUN: llc < %s -mtriple=thumbv7em-linux-gnueabi -mcpu=cortex-m7 | FileCheck %s --check-prefix=CORTEX-M7-DOUBLE
; RUN: llc < %s -mtriple=thumbv7em-linux-gnueabi -mcpu=cortex-m7 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=thumbv8-linux-gnueabi -mcpu=cortex-m23 | FileCheck %s --check-prefix=CORTEX-M23
; RUN: llc < %s -mtriple=thumbv8-linux-gnueabi -mcpu=cortex-m33 | FileCheck %s --check-prefix=CORTEX-M33
; RUN: llc < %s -mtriple=thumbv8-linux-gnueabi -mcpu=cortex-m33 -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-M33-FAST
; RUN: llc < %s -mtriple=thumbv8-linux-gnueabi -mcpu=cortex-m33 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv7r-linux-gnueabi -mcpu=cortex-r4 | FileCheck %s --check-prefix=CORTEX-R4
; RUN: llc < %s -mtriple=armv7r-linux-gnueabi -mcpu=cortex-r4f | FileCheck %s --check-prefix=CORTEX-R4F
; RUN: llc < %s -mtriple=armv7r-linux-gnueabi -mcpu=cortex-r5 | FileCheck %s --check-prefix=CORTEX-R5
; RUN: llc < %s -mtriple=armv7r-linux-gnueabi -mcpu=cortex-r5  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-R5-FAST
; RUN: llc < %s -mtriple=armv7r-linux-gnueabi -mcpu=cortex-r5 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv7r-linux-gnueabi -mcpu=cortex-r7 | FileCheck %s --check-prefix=CORTEX-R7
; RUN: llc < %s -mtriple=armv7r-linux-gnueabi -mcpu=cortex-r7  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-R7-FAST
; RUN: llc < %s -mtriple=armv7r-linux-gnueabi -mcpu=cortex-r7 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv7r-linux-gnueabi -mcpu=cortex-r8 | FileCheck %s --check-prefix=CORTEX-R8
; RUN: llc < %s -mtriple=armv7r-linux-gnueabi -mcpu=cortex-r8  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-R8-FAST
; RUN: llc < %s -mtriple=armv7r-linux-gnueabi -mcpu=cortex-r8 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a32 | FileCheck %s --check-prefix=CORTEX-A32
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a32  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-A32-FAST
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a32 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a35 | FileCheck %s --check-prefix=CORTEX-A35
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a35  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-A35-FAST
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a35 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a53 | FileCheck %s --check-prefix=CORTEX-A53
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a53  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-A53-FAST
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a53 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a57 | FileCheck %s --check-prefix=CORTEX-A57
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a57  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-A57-FAST
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a57 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a72 | FileCheck %s --check-prefix=CORTEX-A72
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a72  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-A72-FAST
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a72 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=cortex-a73 | FileCheck %s --check-prefix=CORTEX-A73
; RUN: llc < %s -mtriple=armv8.1a-linux-gnueabi | FileCheck %s --check-prefix=GENERIC-ARMV8_1-A
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=exynos-m1 | FileCheck %s --check-prefix=EXYNOS-M1
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=exynos-m1  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=EXYNOS-M1-FAST
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=exynos-m1 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=exynos-m2 | FileCheck %s --check-prefix=EXYNOS-M2
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=exynos-m2  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=EXYNOS-M1-FAST
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=exynos-m2 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=exynos-m3 | FileCheck %s --check-prefix=EXYNOS-M3
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=exynos-m3  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=EXYNOS-M1-FAST
; RUN: llc < %s -mtriple=armv8-linux-gnueabi -mcpu=exynos-m3 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv8.1a-linux-gnueabi  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=GENERIC-ARMV8_1-A-FAST
; RUN: llc < %s -mtriple=armv8.1a-linux-gnueabi -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mcpu=cortex-a7 | FileCheck %s  --check-prefix=CORTEX-A7-CHECK
; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mcpu=cortex-a7  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s  --check-prefix=CORTEX-A7-CHECK-FAST
; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mcpu=cortex-a7 -mattr=-vfp2,-vfp3,-vfp4,-neon,-fp16 | FileCheck %s --check-prefix=CORTEX-A7-NOFPU
; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mcpu=cortex-a7 -mattr=-vfp2,-vfp3,-vfp4,-neon,-fp16  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-A7-NOFPU-FAST
; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mcpu=cortex-a7 -mattr=+vfp4,-neon | FileCheck %s --check-prefix=CORTEX-A7-FPUV4
; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mcpu=cortex-a7 -enable-sign-dependent-rounding-fp-math | FileCheck %s --check-prefix=DYN-ROUNDING
; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mcpu=cortex-a7 -mattr=+vfp4,-neon  -enable-unsafe-fp-math -disable-fp-elim -enable-no-infs-fp-math -enable-no-nans-fp-math -fp-contract=fast | FileCheck %s --check-prefix=CORTEX-A7-FPUV4-FAST
; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mcpu=cortex-a7 -mattr=+vfp4,,+d16,-neon | FileCheck %s --check-prefix=CORTEX-A7-FPUV4
; RUN: llc < %s -mtriple=arm-none-linux-gnueabi -mattr=+strict-align -relocation-model=pic | FileCheck %s --check-prefix=RELOC-PIC
; RUN: llc < %s -mtriple=arm-none-linux-gnueabi -mattr=+strict-align -relocation-model=static | FileCheck %s --check-prefix=RELOC-OTHER
; RUN: llc < %s -mtriple=arm-none-linux-gnueabi -mattr=+strict-align -relocation-model=dynamic-no-pic | FileCheck %s --check-prefix=RELOC-OTHER
; RUN: llc < %s -mtriple=arm-none-linux-gnueabi -mattr=+strict-align | FileCheck %s --check-prefix=RELOC-OTHER
; RUN: llc < %s -mtriple=arm-none-linux-gnueabi -mattr=+strict-align | FileCheck %s --check-prefix=PCS-R9-USE
; RUN: llc < %s -mtriple=arm-none-linux-gnueabi -mattr=+reserve-r9,+strict-align | FileCheck %s --check-prefix=PCS-R9-RESERVE
; RUN: llc < %s -mtriple=arm-none-linux-gnueabi -mattr=+strict-align -relocation-model=ropi | FileCheck %s --check-prefix=RELOC-ROPI
; RUN: llc < %s -mtriple=arm-none-linux-gnueabi -mattr=+strict-align -relocation-model=rwpi | FileCheck %s --check-prefix=RELOC-RWPI
; RUN: llc < %s -mtriple=arm-none-linux-gnueabi -mattr=+strict-align -relocation-model=ropi-rwpi | FileCheck %s --check-prefix=RELOC-ROPI-RWPI

; ARMv8.1a (AArch32)
; RUN: llc < %s -mtriple=armv8.1a-none-linux-gnueabi | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; RUN: llc < %s -mtriple=armv8.1a-none-linux-gnueabi -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN
; RUN: llc < %s -mtriple=armv8.1a-none-linux-gnueabi | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; ARMv8a (AArch32)
; RUN: llc < %s -mtriple=armv8-none-linux-gnueabi -mcpu=cortex-a32 | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; RUN: llc < %s -mtriple=armv8-none-linux-gnueabi -mcpu=cortex-a32 -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN
; RUN: llc < %s -mtriple=armv8-none-linux-gnueabi -mcpu=cortex-a35 | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; RUN: llc < %s -mtriple=armv8-none-linux-gnueabi -mcpu=cortex-a35 -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN
; RUN: llc < %s -mtriple=armv8-none-linux-gnueabi -mcpu=cortex-a57 | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; RUN: llc < %s -mtriple=armv8-none-linux-gnueabi -mcpu=cortex-a57 -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN
; RUN: llc < %s -mtriple=armv8-none-linux-gnueabi -mcpu=cortex-a72 | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; RUN: llc < %s -mtriple=armv8-none-linux-gnueabi -mcpu=cortex-a72 -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN
; RUN: llc < %s -mtriple=armv8-none-linux-gnueabi -mcpu=exynos-m1 | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; RUN: llc < %s -mtriple=armv8-none-linux-gnueabi -mcpu=exynos-m1 -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN
; RUN: llc < %s -mtriple=armv8-none-linux-gnueabi -mcpu=exynos-m2 | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; RUN: llc < %s -mtriple=armv8-none-linux-gnueabi -mcpu=exynos-m2 -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN
; RUN: llc < %s -mtriple=armv8-none-linux-gnueabi -mcpu=exynos-m3 | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; RUN: llc < %s -mtriple=armv8-none-linux-gnueabi -mcpu=exynos-m3 -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN

; ARMv7a
; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mcpu=cortex-a7 | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi -mcpu=cortex-a7 -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN
; ARMv7ve
; RUN: llc < %s -mtriple=armv7ve-none-linux-gnueabi | FileCheck %s --check-prefix=V7VE
; ARMv7r
; RUN: llc < %s -mtriple=armv7r-none-linux-gnueabi -mcpu=cortex-r5 | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; RUN: llc < %s -mtriple=armv7r-none-linux-gnueabi -mcpu=cortex-r5 -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN
; ARMv7m
; RUN: llc < %s -mtriple=thumbv7m-none-linux-gnueabi -mcpu=cortex-m3 | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; RUN: llc < %s -mtriple=thumbv7m-none-linux-gnueabi -mcpu=cortex-m3 -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN
; ARMv6
; RUN: llc < %s -mtriple=armv6-none-netbsd-gnueabi -mcpu=arm1136j-s | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; RUN: llc < %s -mtriple=armv6-none-linux-gnueabi -mcpu=arm1136j-s -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN
; RUN: llc < %s -mtriple=armv6-none-linux-gnueabi -mcpu=arm1136j-s | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; ARMv6k
; RUN: llc < %s -mtriple=armv6k-none-netbsd-gnueabi -mcpu=arm1176j-s 2> %t | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; RUN: FileCheck %s < %t --allow-empty --check-prefix=CPU-SUPPORTED
; RUN: llc < %s -mtriple=armv6k-none-linux-gnueabi -mcpu=arm1176j-s -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN
; RUN: llc < %s -mtriple=armv6k-none-linux-gnueabi -mcpu=arm1176j-s | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; ARMv6m
; RUN: llc < %s -mtriple=thumb-none-linux-gnueabi -mcpu=cortex-m0 -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN
; RUN: llc < %s -mtriple=thumb-none-linux-gnueabi -mattr=+strict-align -mcpu=cortex-m0 | FileCheck %s --check-prefix=STRICT-ALIGN
; RUN: llc < %s -mtriple=thumbv6m-none-linux-gnueabi -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN
; RUN: llc < %s -mtriple=thumb-none-linux-gnueabi -mcpu=cortex-m0 -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN
; ARMv5
; RUN: llc < %s -mtriple=armv5-none-linux-gnueabi -mcpu=arm1022e | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; RUN: llc < %s -mtriple=armv5-none-linux-gnueabi -mcpu=arm1022e -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN

; ARMv8-R
; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-r52 -mattr=-vfp2,-fp16 | FileCheck %s --check-prefix=ARMv8R --check-prefix=ARMv8R-NOFPU
; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-r52 -mattr=-neon,+fp-only-sp,+d16 | FileCheck %s --check-prefix=ARMv8R --check-prefix=ARMv8R-SP
; RUN: llc < %s -mtriple=arm-none-none-eabi -mcpu=cortex-r52 | FileCheck %s --check-prefix=ARMv8R --check-prefix=ARMv8R-NEON

; ARMv8-M
; RUN: llc < %s -mtriple=thumbv8-none-none-eabi -mcpu=cortex-m23 | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; RUN: llc < %s -mtriple=thumbv8-none-none-eabi -mcpu=cortex-m23 -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN
; RUN: llc < %s -mtriple=thumbv8-none-none-eabi -mcpu=cortex-m33 | FileCheck %s --check-prefix=NO-STRICT-ALIGN
; RUN: llc < %s -mtriple=thumbv8-none-none-eabi -mcpu=cortex-m33 -mattr=+strict-align | FileCheck %s --check-prefix=STRICT-ALIGN

; CPU-SUPPORTED-NOT: is not a recognized processor for this target

; XSCALE:      .eabi_attribute 6, 5
; XSCALE:      .eabi_attribute 8, 1
; XSCALE:      .eabi_attribute 9, 1

; DYN-ROUNDING: .eabi_attribute 19, 1

; V6:   .eabi_attribute 6, 6
; V6:   .eabi_attribute 8, 1
;; We assume round-to-nearest by default (matches GCC)
; V6-NOT:   .eabi_attribute 27
; V6-NOT:    .eabi_attribute 36
; V6-NOT:    .eabi_attribute 42
; V6-NOT:  .eabi_attribute 44
; V6-NOT:    .eabi_attribute 68
; V6-NOT:   .eabi_attribute 19
;; The default choice made by llc is for a V6 CPU without an FPU.
;; This is not an interesting detail, but for such CPUs, the default intention is to use
;; software floating-point support. The choice is not important for targets without
;; FPU support!
; V6:   .eabi_attribute 20, 1
; V6:   .eabi_attribute 21, 1
; V6-NOT:   .eabi_attribute 22
; V6:   .eabi_attribute 23, 3
; V6:   .eabi_attribute 24, 1
; V6:   .eabi_attribute 25, 1
; V6-NOT:   .eabi_attribute 28
; V6:    .eabi_attribute 38, 1

; V6-FAST-NOT:   .eabi_attribute 19
;; Despite the V6 CPU having no FPU by default, we chose to flush to
;; positive zero here. There's no hardware support doing this, but the
;; fast maths software library might.
; V6-FAST-NOT:   .eabi_attribute 20
; V6-FAST-NOT:   .eabi_attribute 21
; V6-FAST-NOT:   .eabi_attribute 22
; V6-FAST:   .eabi_attribute 23, 1

;; We emit 6, 12 for both v6-M and v6S-M, technically this is incorrect for
;; V6-M, however we don't model the OS extension so this is fine.
; V6M:  .eabi_attribute 6, 12
; V6M:  .eabi_attribute 7, 77
; V6M:  .eabi_attribute 8, 0
; V6M:  .eabi_attribute 9, 1
; V6M-NOT:  .eabi_attribute 27
; V6M-NOT:  .eabi_attribute 36
; V6M-NOT:  .eabi_attribute 42
; V6M-NOT:  .eabi_attribute 44
; V6M-NOT:  .eabi_attribute 68
; V6M-NOT:   .eabi_attribute 19
;; The default choice made by llc is for a V6M CPU without an FPU.
;; This is not an interesting detail, but for such CPUs, the default intention is to use
;; software floating-point support. The choice is not important for targets without
;; FPU support!
; V6M:  .eabi_attribute 20, 1
; V6M:   .eabi_attribute 21, 1
; V6M-NOT:   .eabi_attribute 22
; V6M:   .eabi_attribute 23, 3
; V6M:  .eabi_attribute 24, 1
; V6M:  .eabi_attribute 25, 1
; V6M-NOT:  .eabi_attribute 28
; V6M:  .eabi_attribute 38, 1

; V6M-FAST-NOT:   .eabi_attribute 19
;; Despite the V6M CPU having no FPU by default, we chose to flush to
;; positive zero here. There's no hardware support doing this, but the
;; fast maths software library might.
; V6M-FAST-NOT:  .eabi_attribute 20
; V6M-FAST-NOT:   .eabi_attribute 21
; V6M-FAST-NOT:   .eabi_attribute 22
; V6M-FAST:   .eabi_attribute 23, 1

; ARM1156T2F-S: .cpu arm1156t2f-s
; ARM1156T2F-S: .eabi_attribute 6, 8
; ARM1156T2F-S: .eabi_attribute 8, 1
; ARM1156T2F-S: .eabi_attribute 9, 2
; ARM1156T2F-S: .fpu vfpv2
; ARM1156T2F-S-NOT: .eabi_attribute 27
; ARM1156T2F-S-NOT: .eabi_attribute 36
; ARM1156T2F-S-NOT:    .eabi_attribute 42
; ARM1156T2F-S-NOT:    .eabi_attribute 44
; ARM1156T2F-S-NOT:    .eabi_attribute 68
; ARM1156T2F-S-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; ARM1156T2F-S: .eabi_attribute 20, 1
; ARM1156T2F-S: .eabi_attribute 21, 1
; ARM1156T2F-S-NOT: .eabi_attribute 22
; ARM1156T2F-S: .eabi_attribute 23, 3
; ARM1156T2F-S: .eabi_attribute 24, 1
; ARM1156T2F-S: .eabi_attribute 25, 1
; ARM1156T2F-S-NOT: .eabi_attribute 28
; ARM1156T2F-S: .eabi_attribute 38, 1

; ARM1156T2F-S-FAST-NOT:   .eabi_attribute 19
;; V6 cores default to flush to positive zero (value 0). Note that value 2 is also equally
;; valid for this core, it's an implementation defined question as to which of 0 and 2 you
;; select. LLVM historically picks 0.
; ARM1156T2F-S-FAST-NOT: .eabi_attribute 20
; ARM1156T2F-S-FAST-NOT:   .eabi_attribute 21
; ARM1156T2F-S-FAST-NOT:   .eabi_attribute 22
; ARM1156T2F-S-FAST:   .eabi_attribute 23, 1

; V7M:  .eabi_attribute 6, 10
; V7M:  .eabi_attribute 7, 77
; V7M:  .eabi_attribute 8, 0
; V7M:  .eabi_attribute 9, 2
; V7M-NOT:  .eabi_attribute 27
; V7M-NOT:  .eabi_attribute 36
; V7M-NOT:  .eabi_attribute 42
; V7M-NOT:  .eabi_attribute 44
; V7M-NOT:  .eabi_attribute 68
; V7M-NOT:   .eabi_attribute 19
;; The default choice made by llc is for a V7M CPU without an FPU.
;; This is not an interesting detail, but for such CPUs, the default intention is to use
;; software floating-point support. The choice is not important for targets without
;; FPU support!
; V7M:  .eabi_attribute 20, 1
; V7M: .eabi_attribute 21, 1
; V7M-NOT: .eabi_attribute 22
; V7M: .eabi_attribute 23, 3
; V7M:  .eabi_attribute 24, 1
; V7M:  .eabi_attribute 25, 1
; V7M-NOT:  .eabi_attribute 28
; V7M:  .eabi_attribute 38, 1

; V7M-FAST-NOT:   .eabi_attribute 19
;; Despite the V7M CPU having no FPU by default, we chose to flush
;; preserving sign. This matches what the hardware would do in the
;; architecture revision were to exist on the current target.
; V7M-FAST:  .eabi_attribute 20, 2
; V7M-FAST-NOT:   .eabi_attribute 21
; V7M-FAST-NOT:   .eabi_attribute 22
; V7M-FAST:   .eabi_attribute 23, 1

; V7:      .syntax unified
; V7: .eabi_attribute 6, 10
; V7-NOT: .eabi_attribute 27
; V7-NOT: .eabi_attribute 36
; V7-NOT:    .eabi_attribute 42
; V7-NOT:    .eabi_attribute 44
; V7-NOT:    .eabi_attribute 68
; V7-NOT:   .eabi_attribute 19
;; In safe-maths mode we default to an IEEE 754 compliant choice.
; V7: .eabi_attribute 20, 1
; V7: .eabi_attribute 21, 1
; V7-NOT: .eabi_attribute 22
; V7: .eabi_attribute 23, 3
; V7: .eabi_attribute 24, 1
; V7: .eabi_attribute 25, 1
; V7-NOT: .eabi_attribute 28
; V7: .eabi_attribute 38, 1

; V7-FAST-NOT:   .eabi_attribute 19
;; The default CPU does have an FPU and it must be VFPv3 or better, so it flushes
;; denormals to zero preserving the sign.
; V7-FAST: .eabi_attribute 20, 2
; V7-FAST-NOT:   .eabi_attribute 21
; V7-FAST-NOT:   .eabi_attribute 22
; V7-FAST:   .eabi_attribute 23, 1

; V7VE:      .syntax unified
; V7VE: .eabi_attribute 6, 10   @ Tag_CPU_arch
; V7VE: .eabi_attribute 7, 65   @ Tag_CPU_arch_profile
; V7VE: .eabi_attribute 8, 1    @ Tag_ARM_ISA_use
; V7VE: .eabi_attribute 9, 2    @ Tag_THUMB_ISA_use
; V7VE: .eabi_attribute 42, 1   @ Tag_MPextension_use
; V7VE: .eabi_attribute 44, 2   @ Tag_DIV_use
; V7VE: .eabi_attribute 68, 3   @ Tag_Virtualization_use
; V7VE: .eabi_attribute 17, 1   @ Tag_ABI_PCS_GOT_use
; V7VE: .eabi_attribute 20, 1   @ Tag_ABI_FP_denormal
; V7VE: .eabi_attribute 21, 1   @ Tag_ABI_FP_exceptions
; V7VE: .eabi_attribute 23, 3   @ Tag_ABI_FP_number_model
; V7VE: .eabi_attribute 24, 1   @ Tag_ABI_align_needed
; V7VE: .eabi_attribute 25, 1   @ Tag_ABI_align_preserved
; V7VE: .eabi_attribute 38, 1   @ Tag_ABI_FP_16bit_format

; V8:      .syntax unified
; V8: .eabi_attribute 67, "2.09"
; V8: .eabi_attribute 6, 14
; V8-NOT: .eabi_attribute 44
; V8-NOT:   .eabi_attribute 19
; V8: .eabi_attribute 20, 1
; V8: .eabi_attribute 21, 1
; V8-NOT: .eabi_attribute 22
; V8: .eabi_attribute 23, 3

; V8-FAST-NOT:   .eabi_attribute 19
;; The default does have an FPU, and for V8-A, it flushes preserving sign.
; V8-FAST: .eabi_attribute 20, 2
; V8-FAST-NOT: .eabi_attribute 21
; V8-FAST-NOT: .eabi_attribute 22
; V8-FAST: .eabi_attribute 23, 1

; Vt8:     .syntax unified
; Vt8: .eabi_attribute 6, 14
; Vt8-NOT:   .eabi_attribute 19
; Vt8: .eabi_attribute 20, 1
; Vt8: .eabi_attribute 21, 1
; Vt8-NOT: .eabi_attribute 22
; Vt8: .eabi_attribute 23, 3

; V8-FPARMv8:      .syntax unified
; V8-FPARMv8: .eabi_attribute 6, 14
; V8-FPARMv8: .fpu fp-armv8

; V8-NEON:      .syntax unified
; V8-NEON: .eabi_attribute 6, 14
; V8-NEON: .fpu neon
; V8-NEON: .eabi_attribute 12, 3

; V8-FPARMv8-NEON:      .syntax unified
; V8-FPARMv8-NEON: .eabi_attribute 6, 14
; V8-FPARMv8-NEON: .fpu neon-fp-armv8
; V8-FPARMv8-NEON: .eabi_attribute 12, 3

; V8-FPARMv8-NEON-CRYPTO:      .syntax unified
; V8-FPARMv8-NEON-CRYPTO: .eabi_attribute 6, 14
; V8-FPARMv8-NEON-CRYPTO: .fpu crypto-neon-fp-armv8
; V8-FPARMv8-NEON-CRYPTO: .eabi_attribute 12, 3

; V8MBASELINE: .syntax unified
; '6' is Tag_CPU_arch, '16' is ARM v8-M Baseline
; V8MBASELINE: .eabi_attribute 6, 16
; '7' is Tag_CPU_arch_profile, '77' is 'M'
; V8MBASELINE: .eabi_attribute 7, 77
; '8' is Tag_ARM_ISA_use
; V8MBASELINE: .eabi_attribute 8, 0
; '9' is Tag_Thumb_ISA_use
; V8MBASELINE: .eabi_attribute 9, 3

; V8MMAINLINE: .syntax unified
; '6' is Tag_CPU_arch, '17' is ARM v8-M Mainline
; V8MMAINLINE: .eabi_attribute 6, 17
; V8MMAINLINE: .eabi_attribute 7, 77
; V8MMAINLINE: .eabi_attribute 8, 0
; V8MMAINLINE: .eabi_attribute 9, 3
; V8MMAINLINE_DSP-NOT: .eabi_attribute 46

; V8MMAINLINE_DSP: .syntax unified
; V8MBASELINE_DSP: .eabi_attribute 6, 17
; V8MBASELINE_DSP: .eabi_attribute 7, 77
; V8MMAINLINE_DSP: .eabi_attribute 8, 0
; V8MMAINLINE_DSP: .eabi_attribute 9, 3
; V8MMAINLINE_DSP: .eabi_attribute 46, 1

; Tag_CPU_unaligned_access
; NO-STRICT-ALIGN: .eabi_attribute 34, 1
; STRICT-ALIGN: .eabi_attribute 34, 0

; Tag_CPU_arch  'ARMv7'
; CORTEX-A7-CHECK: .eabi_attribute      6, 10
; CORTEX-A7-NOFPU: .eabi_attribute      6, 10

; CORTEX-A7-FPUV4: .eabi_attribute      6, 10

; Tag_CPU_arch_profile 'A'
; CORTEX-A7-CHECK: .eabi_attribute      7, 65
; CORTEX-A7-NOFPU: .eabi_attribute      7, 65
; CORTEX-A7-FPUV4: .eabi_attribute      7, 65

; Tag_ARM_ISA_use
; CORTEX-A7-CHECK: .eabi_attribute      8, 1
; CORTEX-A7-NOFPU: .eabi_attribute      8, 1
; CORTEX-A7-FPUV4: .eabi_attribute      8, 1

; Tag_THUMB_ISA_use
; CORTEX-A7-CHECK: .eabi_attribute      9, 2
; CORTEX-A7-NOFPU: .eabi_attribute      9, 2
; CORTEX-A7-FPUV4: .eabi_attribute      9, 2

; CORTEX-A7-CHECK: .fpu neon-vfpv4
; CORTEX-A7-NOFPU-NOT: .fpu
; CORTEX-A7-FPUV4: .fpu vfpv4

; CORTEX-A7-CHECK-NOT:   .eabi_attribute 19

; Tag_FP_HP_extension
; CORTEX-A7-CHECK: .eabi_attribute      36, 1
; CORTEX-A7-NOFPU-NOT: .eabi_attribute  36
; CORTEX-A7-FPUV4: .eabi_attribute      36, 1

; Tag_MPextension_use
; CORTEX-A7-CHECK: .eabi_attribute      42, 1
; CORTEX-A7-NOFPU: .eabi_attribute      42, 1
; CORTEX-A7-FPUV4: .eabi_attribute      42, 1

; Tag_DIV_use
; CORTEX-A7-CHECK: .eabi_attribute      44, 2
; CORTEX-A7-NOFPU: .eabi_attribute      44, 2
; CORTEX-A7-FPUV4: .eabi_attribute      44, 2

; Tag_DSP_extension
; CORTEX-A7-CHECK-NOT: .eabi_attribute      46

; Tag_Virtualization_use
; CORTEX-A7-CHECK: .eabi_attribute      68, 3
; CORTEX-A7-NOFPU: .eabi_attribute      68, 3
; CORTEX-A7-FPUV4: .eabi_attribute      68, 3

; Tag_ABI_FP_denormal
;; We default to IEEE 754 compliance
; CORTEX-A7-CHECK: .eabi_attribute      20, 1
;; The A7 has VFPv3 support by default, so flush preserving sign.
; CORTEX-A7-CHECK-FAST: .eabi_attribute 20, 2
; CORTEX-A7-NOFPU: .eabi_attribute      20, 1
;; Despite there being no FPU, we chose to flush to zero preserving
;; sign. This matches what the hardware would do for this architecture
;; revision.
; CORTEX-A7-NOFPU-FAST: .eabi_attribute 20, 2
; CORTEX-A7-FPUV4: .eabi_attribute      20, 1
;; The VFPv4 FPU flushes preserving sign.
; CORTEX-A7-FPUV4-FAST: .eabi_attribute 20, 2

; Tag_ABI_FP_exceptions
; CORTEX-A7-CHECK: .eabi_attribute      21, 1
; CORTEX-A7-NOFPU: .eabi_attribute      21, 1
; CORTEX-A7-FPUV4: .eabi_attribute      21, 1

; Tag_ABI_FP_user_exceptions
; CORTEX-A7-CHECK-NOT: .eabi_attribute      22
; CORTEX-A7-NOFPU-NOT: .eabi_attribute      22
; CORTEX-A7-FPUV4-NOT: .eabi_attribute      22

; Tag_ABI_FP_number_model
; CORTEX-A7-CHECK: .eabi_attribute      23, 3
; CORTEX-A7-NOFPU: .eabi_attribute      23, 3
; CORTEX-A7-FPUV4: .eabi_attribute      23, 3

; Tag_ABI_align_needed
; CORTEX-A7-CHECK: .eabi_attribute      24, 1
; CORTEX-A7-NOFPU: .eabi_attribute      24, 1
; CORTEX-A7-FPUV4: .eabi_attribute      24, 1

; Tag_ABI_align_preserved
; CORTEX-A7-CHECK: .eabi_attribute      25, 1
; CORTEX-A7-NOFPU: .eabi_attribute      25, 1
; CORTEX-A7-FPUV4: .eabi_attribute      25, 1

; Tag_FP_16bit_format
; CORTEX-A7-CHECK: .eabi_attribute      38, 1
; CORTEX-A7-NOFPU: .eabi_attribute      38, 1
; CORTEX-A7-FPUV4: .eabi_attribute      38, 1

; CORTEX-A5-DEFAULT:        .cpu    cortex-a5
; CORTEX-A5-DEFAULT:        .eabi_attribute 6, 10
; CORTEX-A5-DEFAULT:        .eabi_attribute 7, 65
; CORTEX-A5-DEFAULT:        .eabi_attribute 8, 1
; CORTEX-A5-DEFAULT:        .eabi_attribute 9, 2
; CORTEX-A5-DEFAULT:        .fpu    neon-vfpv4
; CORTEX-A5-DEFAULT:        .eabi_attribute 42, 1
; CORTEX-A5-DEFAULT-NOT:        .eabi_attribute 44
; CORTEX-A5-DEFAULT:        .eabi_attribute 68, 1
; CORTEX-A5-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-A5-DEFAULT:        .eabi_attribute 20, 1
; CORTEX-A5-DEFAULT:        .eabi_attribute 21, 1
; CORTEX-A5-DEFAULT-NOT:        .eabi_attribute 22
; CORTEX-A5-DEFAULT:        .eabi_attribute 23, 3
; CORTEX-A5-DEFAULT:        .eabi_attribute 24, 1
; CORTEX-A5-DEFAULT:        .eabi_attribute 25, 1

; CORTEX-A5-DEFAULT-FAST-NOT:   .eabi_attribute 19
;; The A5 defaults to a VFPv4 FPU, so it flushed preserving the sign when -ffast-math
;; is given.
; CORTEX-A5-DEFAULT-FAST:        .eabi_attribute 20, 2
; CORTEX-A5-DEFAULT-FAST-NOT: .eabi_attribute 21
; CORTEX-A5-DEFAULT-FAST-NOT: .eabi_attribute 22
; CORTEX-A5-DEFAULT-FAST: .eabi_attribute 23, 1

; CORTEX-A5-NONEON:        .cpu    cortex-a5
; CORTEX-A5-NONEON:        .eabi_attribute 6, 10
; CORTEX-A5-NONEON:        .eabi_attribute 7, 65
; CORTEX-A5-NONEON:        .eabi_attribute 8, 1
; CORTEX-A5-NONEON:        .eabi_attribute 9, 2
; CORTEX-A5-NONEON:        .fpu    vfpv4-d16
; CORTEX-A5-NONEON:        .eabi_attribute 42, 1
; CORTEX-A5-NONEON:        .eabi_attribute 68, 1
;; We default to IEEE 754 compliance
; CORTEX-A5-NONEON:        .eabi_attribute 20, 1
; CORTEX-A5-NONEON:        .eabi_attribute 21, 1
; CORTEX-A5-NONEON-NOT:    .eabi_attribute 22
; CORTEX-A5-NONEON:        .eabi_attribute 23, 3
; CORTEX-A5-NONEON:        .eabi_attribute 24, 1
; CORTEX-A5-NONEON:        .eabi_attribute 25, 1

; CORTEX-A5-NONEON-FAST-NOT:   .eabi_attribute 19
;; The A5 defaults to a VFPv4 FPU, so it flushed preserving sign when -ffast-math
;; is given.
; CORTEX-A5-NONEON-FAST:        .eabi_attribute 20, 2
; CORTEX-A5-NONEON-FAST-NOT: .eabi_attribute 21
; CORTEX-A5-NONEON-FAST-NOT: .eabi_attribute 22
; CORTEX-A5-NONEON-FAST: .eabi_attribute 23, 1

; CORTEX-A5-NOFPU:        .cpu    cortex-a5
; CORTEX-A5-NOFPU:        .eabi_attribute 6, 10
; CORTEX-A5-NOFPU:        .eabi_attribute 7, 65
; CORTEX-A5-NOFPU:        .eabi_attribute 8, 1
; CORTEX-A5-NOFPU:        .eabi_attribute 9, 2
; CORTEX-A5-NOFPU-NOT:    .fpu
; CORTEX-A5-NOFPU:        .eabi_attribute 42, 1
; CORTEX-A5-NOFPU:        .eabi_attribute 68, 1
; CORTEX-A5-NOFPU-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-A5-NOFPU:        .eabi_attribute 20, 1
; CORTEX-A5-NOFPU:        .eabi_attribute 21, 1
; CORTEX-A5-NOFPU-NOT:    .eabi_attribute 22
; CORTEX-A5-NOFPU:        .eabi_attribute 23, 3
; CORTEX-A5-NOFPU:        .eabi_attribute 24, 1
; CORTEX-A5-NOFPU:        .eabi_attribute 25, 1

; CORTEX-A5-NOFPU-FAST-NOT:   .eabi_attribute 19
;; Despite there being no FPU, we chose to flush to zero preserving
;; sign. This matches what the hardware would do for this architecture
;; revision.
; CORTEX-A5-NOFPU-FAST: .eabi_attribute 20, 2
; CORTEX-A5-NOFPU-FAST-NOT: .eabi_attribute 21
; CORTEX-A5-NOFPU-FAST-NOT: .eabi_attribute 22
; CORTEX-A5-NOFPU-FAST: .eabi_attribute 23, 1

; CORTEX-A8-SOFT:  .cpu cortex-a8
; CORTEX-A8-SOFT:  .eabi_attribute 6, 10
; CORTEX-A8-SOFT:  .eabi_attribute 7, 65
; CORTEX-A8-SOFT:  .eabi_attribute 8, 1
; CORTEX-A8-SOFT:  .eabi_attribute 9, 2
; CORTEX-A8-SOFT:  .fpu neon
; CORTEX-A8-SOFT-NOT:  .eabi_attribute 27
; CORTEX-A8-SOFT-NOT:  .eabi_attribute 36, 1
; CORTEX-A8-SOFT-NOT:  .eabi_attribute 42, 1
; CORTEX-A8-SOFT-NOT:  .eabi_attribute 44
; CORTEX-A8-SOFT:  .eabi_attribute 68, 1
; CORTEX-A8-SOFT-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-A8-SOFT:  .eabi_attribute 20, 1
; CORTEX-A8-SOFT:  .eabi_attribute 21, 1
; CORTEX-A8-SOFT-NOT:  .eabi_attribute 22
; CORTEX-A8-SOFT:  .eabi_attribute 23, 3
; CORTEX-A8-SOFT:  .eabi_attribute 24, 1
; CORTEX-A8-SOFT:  .eabi_attribute 25, 1
; CORTEX-A8-SOFT-NOT:  .eabi_attribute 28
; CORTEX-A8-SOFT:  .eabi_attribute 38, 1

; CORTEX-A9-SOFT:  .cpu cortex-a9
; CORTEX-A9-SOFT:  .eabi_attribute 6, 10
; CORTEX-A9-SOFT:  .eabi_attribute 7, 65
; CORTEX-A9-SOFT:  .eabi_attribute 8, 1
; CORTEX-A9-SOFT:  .eabi_attribute 9, 2
; CORTEX-A9-SOFT:  .fpu neon
; CORTEX-A9-SOFT-NOT:  .eabi_attribute 27
; CORTEX-A9-SOFT:  .eabi_attribute 36, 1
; CORTEX-A9-SOFT:  .eabi_attribute 42, 1
; CORTEX-A9-SOFT-NOT:  .eabi_attribute 44
; CORTEX-A9-SOFT:  .eabi_attribute 68, 1
; CORTEX-A9-SOFT-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-A9-SOFT:  .eabi_attribute 20, 1
; CORTEX-A9-SOFT:  .eabi_attribute 21, 1
; CORTEX-A9-SOFT-NOT:  .eabi_attribute 22
; CORTEX-A9-SOFT:  .eabi_attribute 23, 3
; CORTEX-A9-SOFT:  .eabi_attribute 24, 1
; CORTEX-A9-SOFT:  .eabi_attribute 25, 1
; CORTEX-A9-SOFT-NOT:  .eabi_attribute 28
; CORTEX-A9-SOFT:  .eabi_attribute 38, 1

; CORTEX-A8-SOFT-FAST-NOT:   .eabi_attribute 19
; CORTEX-A9-SOFT-FAST-NOT:   .eabi_attribute 19
;; The A9 defaults to a VFPv3 FPU, so it flushes preserving the sign when
;; -ffast-math is specified.
; CORTEX-A8-SOFT-FAST:  .eabi_attribute 20, 2
; CORTEX-A9-SOFT-FAST:  .eabi_attribute 20, 2
; CORTEX-A5-SOFT-FAST-NOT: .eabi_attribute 21
; CORTEX-A5-SOFT-FAST-NOT: .eabi_attribute 22
; CORTEX-A5-SOFT-FAST: .eabi_attribute 23, 1

; CORTEX-A8-HARD:  .cpu cortex-a8
; CORTEX-A8-HARD:  .eabi_attribute 6, 10
; CORTEX-A8-HARD:  .eabi_attribute 7, 65
; CORTEX-A8-HARD:  .eabi_attribute 8, 1
; CORTEX-A8-HARD:  .eabi_attribute 9, 2
; CORTEX-A8-HARD:  .fpu neon
; CORTEX-A8-HARD-NOT:  .eabi_attribute 27
; CORTEX-A8-HARD-NOT:  .eabi_attribute 36, 1
; CORTEX-A8-HARD-NOT:  .eabi_attribute 42, 1
; CORTEX-A8-HARD:  .eabi_attribute 68, 1
; CORTEX-A8-HARD-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-A8-HARD:  .eabi_attribute 20, 1
; CORTEX-A8-HARD:  .eabi_attribute 21, 1
; CORTEX-A8-HARD-NOT:  .eabi_attribute 22
; CORTEX-A8-HARD:  .eabi_attribute 23, 3
; CORTEX-A8-HARD:  .eabi_attribute 24, 1
; CORTEX-A8-HARD:  .eabi_attribute 25, 1
; CORTEX-A8-HARD:  .eabi_attribute 28, 1
; CORTEX-A8-HARD:  .eabi_attribute 38, 1



; CORTEX-A9-HARD:  .cpu cortex-a9
; CORTEX-A9-HARD:  .eabi_attribute 6, 10
; CORTEX-A9-HARD:  .eabi_attribute 7, 65
; CORTEX-A9-HARD:  .eabi_attribute 8, 1
; CORTEX-A9-HARD:  .eabi_attribute 9, 2
; CORTEX-A9-HARD:  .fpu neon
; CORTEX-A9-HARD-NOT:  .eabi_attribute 27
; CORTEX-A9-HARD:  .eabi_attribute 36, 1
; CORTEX-A9-HARD:  .eabi_attribute 42, 1
; CORTEX-A9-HARD:  .eabi_attribute 68, 1
; CORTEX-A9-HARD-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-A9-HARD:  .eabi_attribute 20, 1
; CORTEX-A9-HARD:  .eabi_attribute 21, 1
; CORTEX-A9-HARD-NOT:  .eabi_attribute 22
; CORTEX-A9-HARD:  .eabi_attribute 23, 3
; CORTEX-A9-HARD:  .eabi_attribute 24, 1
; CORTEX-A9-HARD:  .eabi_attribute 25, 1
; CORTEX-A9-HARD:  .eabi_attribute 28, 1
; CORTEX-A9-HARD:  .eabi_attribute 38, 1

; CORTEX-A8-HARD-FAST-NOT:   .eabi_attribute 19
;; The A8 defaults to a VFPv3 FPU, so it flushes preserving the sign when
;; -ffast-math is specified.
; CORTEX-A8-HARD-FAST:  .eabi_attribute 20, 2
; CORTEX-A8-HARD-FAST-NOT:  .eabi_attribute 21
; CORTEX-A8-HARD-FAST-NOT:  .eabi_attribute 22
; CORTEX-A8-HARD-FAST:  .eabi_attribute 23, 1

; CORTEX-A9-HARD-FAST-NOT:   .eabi_attribute 19
;; The A9 defaults to a VFPv3 FPU, so it flushes preserving the sign when
;; -ffast-math is specified.
; CORTEX-A9-HARD-FAST:  .eabi_attribute 20, 2
; CORTEX-A9-HARD-FAST-NOT:  .eabi_attribute 21
; CORTEX-A9-HARD-FAST-NOT:  .eabi_attribute 22
; CORTEX-A9-HARD-FAST:  .eabi_attribute 23, 1

; CORTEX-A12-DEFAULT:  .cpu cortex-a12
; CORTEX-A12-DEFAULT:  .eabi_attribute 6, 10
; CORTEX-A12-DEFAULT:  .eabi_attribute 7, 65
; CORTEX-A12-DEFAULT:  .eabi_attribute 8, 1
; CORTEX-A12-DEFAULT:  .eabi_attribute 9, 2
; CORTEX-A12-DEFAULT:  .fpu neon-vfpv4
; CORTEX-A12-DEFAULT:  .eabi_attribute 42, 1
; CORTEX-A12-DEFAULT:  .eabi_attribute 44, 2
; CORTEX-A12-DEFAULT:  .eabi_attribute 68, 3
; CORTEX-A12-DEFAULT-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-A12-DEFAULT:  .eabi_attribute 20, 1
; CORTEX-A12-DEFAULT:  .eabi_attribute 21, 1
; CORTEX-A12-DEFAULT-NOT:  .eabi_attribute 22
; CORTEX-A12-DEFAULT:  .eabi_attribute 23, 3
; CORTEX-A12-DEFAULT:  .eabi_attribute 24, 1
; CORTEX-A12-DEFAULT:  .eabi_attribute 25, 1

; CORTEX-A12-DEFAULT-FAST-NOT:   .eabi_attribute 19
;; The A12 defaults to a VFPv3 FPU, so it flushes preserving the sign when
;; -ffast-math is specified.
; CORTEX-A12-DEFAULT-FAST:  .eabi_attribute 20, 2
; CORTEX-A12-HARD-FAST-NOT:  .eabi_attribute 21
; CORTEX-A12-HARD-FAST-NOT:  .eabi_attribute 22
; CORTEX-A12-HARD-FAST:  .eabi_attribute 23, 1

; CORTEX-A12-NOFPU:  .cpu cortex-a12
; CORTEX-A12-NOFPU:  .eabi_attribute 6, 10
; CORTEX-A12-NOFPU:  .eabi_attribute 7, 65
; CORTEX-A12-NOFPU:  .eabi_attribute 8, 1
; CORTEX-A12-NOFPU:  .eabi_attribute 9, 2
; CORTEX-A12-NOFPU-NOT:  .fpu
; CORTEX-A12-NOFPU:  .eabi_attribute 42, 1
; CORTEX-A12-NOFPU:  .eabi_attribute 44, 2
; CORTEX-A12-NOFPU:  .eabi_attribute 68, 3
; CORTEX-A12-NOFPU-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-A12-NOFPU:  .eabi_attribute 20, 1
; CORTEX-A12-NOFPU:  .eabi_attribute 21, 1
; CORTEX-A12-NOFPU-NOT:  .eabi_attribute 22
; CORTEX-A12-NOFPU:  .eabi_attribute 23, 3
; CORTEX-A12-NOFPU:  .eabi_attribute 24, 1
; CORTEX-A12-NOFPU:  .eabi_attribute 25, 1

; CORTEX-A12-NOFPU-FAST-NOT:   .eabi_attribute 19
;; Despite there being no FPU, we chose to flush to zero preserving
;; sign. This matches what the hardware would do for this architecture
;; revision.
; CORTEX-A12-NOFPU-FAST:  .eabi_attribute 20, 2
; CORTEX-A12-NOFPU-FAST-NOT:  .eabi_attribute 21
; CORTEX-A12-NOFPU-FAST-NOT:  .eabi_attribute 22
; CORTEX-A12-NOFPU-FAST:  .eabi_attribute 23, 1

; CORTEX-A15: .cpu cortex-a15
; CORTEX-A15: .eabi_attribute 6, 10
; CORTEX-A15: .eabi_attribute 7, 65
; CORTEX-A15: .eabi_attribute 8, 1
; CORTEX-A15: .eabi_attribute 9, 2
; CORTEX-A15: .fpu neon-vfpv4
; CORTEX-A15-NOT: .eabi_attribute 27
; CORTEX-A15: .eabi_attribute 36, 1
; CORTEX-A15: .eabi_attribute 42, 1
; CORTEX-A15: .eabi_attribute 44, 2
; CORTEX-A15: .eabi_attribute 68, 3
; CORTEX-A15-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-A15: .eabi_attribute 20, 1
; CORTEX-A15: .eabi_attribute 21, 1
; CORTEX-A15-NOT: .eabi_attribute 22
; CORTEX-A15: .eabi_attribute 23, 3
; CORTEX-A15: .eabi_attribute 24, 1
; CORTEX-A15: .eabi_attribute 25, 1
; CORTEX-A15-NOT: .eabi_attribute 28
; CORTEX-A15: .eabi_attribute 38, 1

; CORTEX-A15-FAST-NOT:   .eabi_attribute 19
;; The A15 defaults to a VFPv3 FPU, so it flushes preserving the sign when
;; -ffast-math is specified.
; CORTEX-A15-FAST: .eabi_attribute 20, 2
; CORTEX-A15-FAST-NOT:  .eabi_attribute 21
; CORTEX-A15-FAST-NOT:  .eabi_attribute 22
; CORTEX-A15-FAST:  .eabi_attribute 23, 1

; CORTEX-A17-DEFAULT:  .cpu cortex-a17
; CORTEX-A17-DEFAULT:  .eabi_attribute 6, 10
; CORTEX-A17-DEFAULT:  .eabi_attribute 7, 65
; CORTEX-A17-DEFAULT:  .eabi_attribute 8, 1
; CORTEX-A17-DEFAULT:  .eabi_attribute 9, 2
; CORTEX-A17-DEFAULT:  .fpu neon-vfpv4
; CORTEX-A17-DEFAULT:  .eabi_attribute 42, 1
; CORTEX-A17-DEFAULT:  .eabi_attribute 44, 2
; CORTEX-A17-DEFAULT:  .eabi_attribute 68, 3
; CORTEX-A17-DEFAULT-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-A17-DEFAULT:  .eabi_attribute 20, 1
; CORTEX-A17-DEFAULT:  .eabi_attribute 21, 1
; CORTEX-A17-DEFAULT-NOT:  .eabi_attribute 22
; CORTEX-A17-DEFAULT:  .eabi_attribute 23, 3
; CORTEX-A17-DEFAULT:  .eabi_attribute 24, 1
; CORTEX-A17-DEFAULT:  .eabi_attribute 25, 1

; CORTEX-A17-FAST-NOT:   .eabi_attribute 19
;; The A17 defaults to a VFPv3 FPU, so it flushes preserving the sign when
;; -ffast-math is specified.
; CORTEX-A17-FAST:  .eabi_attribute 20, 2
; CORTEX-A17-FAST-NOT:  .eabi_attribute 21
; CORTEX-A17-FAST-NOT:  .eabi_attribute 22
; CORTEX-A17-FAST:  .eabi_attribute 23, 1

; CORTEX-A17-NOFPU:  .cpu cortex-a17
; CORTEX-A17-NOFPU:  .eabi_attribute 6, 10
; CORTEX-A17-NOFPU:  .eabi_attribute 7, 65
; CORTEX-A17-NOFPU:  .eabi_attribute 8, 1
; CORTEX-A17-NOFPU:  .eabi_attribute 9, 2
; CORTEX-A17-NOFPU-NOT:  .fpu
; CORTEX-A17-NOFPU:  .eabi_attribute 42, 1
; CORTEX-A17-NOFPU:  .eabi_attribute 44, 2
; CORTEX-A17-NOFPU:  .eabi_attribute 68, 3
; CORTEX-A17-NOFPU-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-A17-NOFPU:  .eabi_attribute 20, 1
; CORTEX-A17-NOFPU:  .eabi_attribute 21, 1
; CORTEX-A17-NOFPU-NOT:  .eabi_attribute 22
; CORTEX-A17-NOFPU:  .eabi_attribute 23, 3
; CORTEX-A17-NOFPU:  .eabi_attribute 24, 1
; CORTEX-A17-NOFPU:  .eabi_attribute 25, 1

; CORTEX-A17-NOFPU-NOT:   .eabi_attribute 19
;; Despite there being no FPU, we chose to flush to zero preserving
;; sign. This matches what the hardware would do for this architecture
;; revision.
; CORTEX-A17-NOFPU-FAST:  .eabi_attribute 20, 2
; CORTEX-A17-NOFPU-FAST-NOT:  .eabi_attribute 21
; CORTEX-A17-NOFPU-FAST-NOT:  .eabi_attribute 22
; CORTEX-A17-NOFPU-FAST:  .eabi_attribute 23, 1

; Test flags -enable-no-trapping-fp-math and -denormal-fp-math:
; NO-TRAPPING-MATH:  .eabi_attribute 21, 0
; DENORMAL-IEEE:  .eabi_attribute 20, 1
; DENORMAL-PRESERVE-SIGN:  .eabi_attribute 20, 2
; DENORMAL-POSITIVE-ZERO:  .eabi_attribute 20, 0

; CORTEX-M0:  .cpu cortex-m0
; CORTEX-M0:  .eabi_attribute 6, 12
; CORTEX-M0:  .eabi_attribute 7, 77
; CORTEX-M0:  .eabi_attribute 8, 0
; CORTEX-M0:  .eabi_attribute 9, 1
; CORTEX-M0-NOT:  .eabi_attribute 27
; CORTEX-M0-NOT:  .eabi_attribute 36
; CORTEX-M0: .eabi_attribute 34, 0
; CORTEX-M0-NOT:  .eabi_attribute 42
; CORTEX-M0-NOT:  .eabi_attribute 44
; CORTEX-M0-NOT:  .eabi_attribute 68
; CORTEX-M0-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-M0:  .eabi_attribute 20, 1
; CORTEX-M0:  .eabi_attribute 21, 1
; CORTEX-M0-NOT:  .eabi_attribute 22
; CORTEX-M0:  .eabi_attribute 23, 3
; CORTEX-M0:  .eabi_attribute 24, 1
; CORTEX-M0:  .eabi_attribute 25, 1
; CORTEX-M0-NOT:  .eabi_attribute 28
; CORTEX-M0:  .eabi_attribute 38, 1

; CORTEX-M0-FAST-NOT:   .eabi_attribute 19
;; Despite the M0 CPU having no FPU in this scenario, we chose to
;; flush to positive zero here. There's no hardware support doing
;; this, but the fast maths software library might and such behaviour
;; would match hardware support on this architecture revision if it
;; existed.
; CORTEX-M0-FAST-NOT:  .eabi_attribute 20
; CORTEX-M0-FAST-NOT:  .eabi_attribute 21
; CORTEX-M0-FAST-NOT:  .eabi_attribute 22
; CORTEX-M0-FAST:  .eabi_attribute 23, 1

; CORTEX-M0PLUS:  .cpu cortex-m0plus
; CORTEX-M0PLUS:  .eabi_attribute 6, 12
; CORTEX-M0PLUS:  .eabi_attribute 7, 77
; CORTEX-M0PLUS:  .eabi_attribute 8, 0
; CORTEX-M0PLUS:  .eabi_attribute 9, 1
; CORTEX-M0PLUS-NOT:  .eabi_attribute 27
; CORTEX-M0PLUS-NOT:  .eabi_attribute 36
; CORTEX-M0PLUS-NOT:  .eabi_attribute 42
; CORTEX-M0PLUS-NOT:  .eabi_attribute 44
; CORTEX-M0PLUS-NOT:  .eabi_attribute 68
; CORTEX-M0PLUS-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-M0PLUS:  .eabi_attribute 20, 1
; CORTEX-M0PLUS:  .eabi_attribute 21, 1
; CORTEX-M0PLUS-NOT:  .eabi_attribute 22
; CORTEX-M0PLUS:  .eabi_attribute 23, 3
; CORTEX-M0PLUS:  .eabi_attribute 24, 1
; CORTEX-M0PLUS:  .eabi_attribute 25, 1
; CORTEX-M0PLUS-NOT:  .eabi_attribute 28
; CORTEX-M0PLUS:  .eabi_attribute 38, 1

; CORTEX-M0PLUS-FAST-NOT:   .eabi_attribute 19
;; Despite the M0+ CPU having no FPU in this scenario, we chose to
;; flush to positive zero here. There's no hardware support doing
;; this, but the fast maths software library might and such behaviour
;; would match hardware support on this architecture revision if it
;; existed.
; CORTEX-M0PLUS-FAST-NOT:  .eabi_attribute 20
; CORTEX-M0PLUS-FAST-NOT:  .eabi_attribute 21
; CORTEX-M0PLUS-FAST-NOT:  .eabi_attribute 22
; CORTEX-M0PLUS-FAST:  .eabi_attribute 23, 1

; CORTEX-M1:  .cpu cortex-m1
; CORTEX-M1:  .eabi_attribute 6, 12
; CORTEX-M1:  .eabi_attribute 7, 77
; CORTEX-M1:  .eabi_attribute 8, 0
; CORTEX-M1:  .eabi_attribute 9, 1
; CORTEX-M1-NOT:  .eabi_attribute 27
; CORTEX-M1-NOT:  .eabi_attribute 36
; CORTEX-M1-NOT:  .eabi_attribute 42
; CORTEX-M1-NOT:  .eabi_attribute 44
; CORTEX-M1-NOT:  .eabi_attribute 68
; CORTEX-M1-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-M1:  .eabi_attribute 20, 1
; CORTEX-M1:  .eabi_attribute 21, 1
; CORTEX-M1-NOT:  .eabi_attribute 22
; CORTEX-M1:  .eabi_attribute 23, 3
; CORTEX-M1:  .eabi_attribute 24, 1
; CORTEX-M1:  .eabi_attribute 25, 1
; CORTEX-M1-NOT:  .eabi_attribute 28
; CORTEX-M1:  .eabi_attribute 38, 1

; CORTEX-M1-FAST-NOT:   .eabi_attribute 19
;; Despite the M1 CPU having no FPU in this scenario, we chose to
;; flush to positive zero here. There's no hardware support doing
;; this, but the fast maths software library might and such behaviour
;; would match hardware support on this architecture revision if it
;; existed.
; CORTEX-M1-FAST-NOT:  .eabi_attribute 20
; CORTEX-M1-FAST-NOT:  .eabi_attribute 21
; CORTEX-M1-FAST-NOT:  .eabi_attribute 22
; CORTEX-M1-FAST:  .eabi_attribute 23, 1

; SC000:  .cpu sc000
; SC000:  .eabi_attribute 6, 12
; SC000:  .eabi_attribute 7, 77
; SC000:  .eabi_attribute 8, 0
; SC000:  .eabi_attribute 9, 1
; SC000-NOT:  .eabi_attribute 27
; SC000-NOT:  .eabi_attribute 42
; SC000-NOT:  .eabi_attribute 44
; SC000-NOT:  .eabi_attribute 68
; SC000-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; SC000:  .eabi_attribute 20, 1
; SC000:  .eabi_attribute 21, 1
; SC000-NOT:  .eabi_attribute 22
; SC000:  .eabi_attribute 23, 3
; SC000:  .eabi_attribute 24, 1
; SC000:  .eabi_attribute 25, 1
; SC000-NOT:  .eabi_attribute 28
; SC000:  .eabi_attribute 38, 1

; SC000-FAST-NOT:   .eabi_attribute 19
;; Despite the SC000 CPU having no FPU in this scenario, we chose to
;; flush to positive zero here. There's no hardware support doing
;; this, but the fast maths software library might and such behaviour
;; would match hardware support on this architecture revision if it
;; existed.
; SC000-FAST-NOT:  .eabi_attribute 20
; SC000-FAST-NOT:  .eabi_attribute 21
; SC000-FAST-NOT:  .eabi_attribute 22
; SC000-FAST:  .eabi_attribute 23, 1

; CORTEX-M3:  .cpu cortex-m3
; CORTEX-M3:  .eabi_attribute 6, 10
; CORTEX-M3:  .eabi_attribute 7, 77
; CORTEX-M3:  .eabi_attribute 8, 0
; CORTEX-M3:  .eabi_attribute 9, 2
; CORTEX-M3-NOT:  .eabi_attribute 27
; CORTEX-M3-NOT:  .eabi_attribute 36
; CORTEX-M3-NOT:  .eabi_attribute 42
; CORTEX-M3-NOT:  .eabi_attribute 44
; CORTEX-M3-NOT:  .eabi_attribute 68
; CORTEX-M3-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-M3:  .eabi_attribute 20, 1
; CORTEX-M3:  .eabi_attribute 21, 1
; CORTEX-M3-NOT:  .eabi_attribute 22
; CORTEX-M3:  .eabi_attribute 23, 3
; CORTEX-M3:  .eabi_attribute 24, 1
; CORTEX-M3:  .eabi_attribute 25, 1
; CORTEX-M3-NOT:  .eabi_attribute 28
; CORTEX-M3:  .eabi_attribute 38, 1

; CORTEX-M3-FAST-NOT:   .eabi_attribute 19
;; Despite there being no FPU, we chose to flush to zero preserving
;; sign. This matches what the hardware would do for this architecture
;; revision.
; CORTEX-M3-FAST:  .eabi_attribute 20, 2
; CORTEX-M3-FAST-NOT:  .eabi_attribute 21
; CORTEX-M3-FAST-NOT:  .eabi_attribute 22
; CORTEX-M3-FAST:  .eabi_attribute 23, 1

; SC300:  .cpu sc300
; SC300:  .eabi_attribute 6, 10
; SC300:  .eabi_attribute 7, 77
; SC300:  .eabi_attribute 8, 0
; SC300:  .eabi_attribute 9, 2
; SC300-NOT:  .eabi_attribute 27
; SC300-NOT:  .eabi_attribute 36
; SC300-NOT:  .eabi_attribute 42
; SC300-NOT:  .eabi_attribute 44
; SC300-NOT:  .eabi_attribute 68
; SC300-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; SC300:  .eabi_attribute 20, 1
; SC300:  .eabi_attribute 21, 1
; SC300-NOT:  .eabi_attribute 22
; SC300:  .eabi_attribute 23, 3
; SC300:  .eabi_attribute 24, 1
; SC300:  .eabi_attribute 25, 1
; SC300-NOT:  .eabi_attribute 28
; SC300:  .eabi_attribute 38, 1

; SC300-FAST-NOT:   .eabi_attribute 19
;; Despite there being no FPU, we chose to flush to zero preserving
;; sign. This matches what the hardware would do for this architecture
;; revision.
; SC300-FAST:  .eabi_attribute 20, 2
; SC300-FAST-NOT:  .eabi_attribute 21
; SC300-FAST-NOT:  .eabi_attribute 22
; SC300-FAST:  .eabi_attribute 23, 1

; CORTEX-M4-SOFT:  .cpu cortex-m4
; CORTEX-M4-SOFT:  .eabi_attribute 6, 13
; CORTEX-M4-SOFT:  .eabi_attribute 7, 77
; CORTEX-M4-SOFT:  .eabi_attribute 8, 0
; CORTEX-M4-SOFT:  .eabi_attribute 9, 2
; CORTEX-M4-SOFT:  .fpu fpv4-sp-d16
; CORTEX-M4-SOFT:  .eabi_attribute 27, 1
; CORTEX-M4-SOFT:  .eabi_attribute 36, 1
; CORTEX-M4-SOFT-NOT:  .eabi_attribute 42
; CORTEX-M4-SOFT-NOT:  .eabi_attribute 44
; CORTEX-M4-SOFT-NOT:  .eabi_attribute 68
; CORTEX-M4-SOFT-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-M4-SOFT:  .eabi_attribute 20, 1
; CORTEX-M4-SOFT:  .eabi_attribute 21, 1
; CORTEX-M4-SOFT-NOT:  .eabi_attribute 22
; CORTEX-M4-SOFT:  .eabi_attribute 23, 3
; CORTEX-M4-SOFT:  .eabi_attribute 24, 1
; CORTEX-M4-SOFT:  .eabi_attribute 25, 1
; CORTEX-M4-SOFT-NOT:  .eabi_attribute 28
; CORTEX-M4-SOFT:  .eabi_attribute 38, 1

; CORTEX-M4-SOFT-FAST-NOT:   .eabi_attribute 19
;; The M4 defaults to a VFPv4 FPU, so it flushes preserving the sign when
;; -ffast-math is specified.
; CORTEX-M4-SOFT-FAST:  .eabi_attribute 20, 2
; CORTEX-M4-SOFT-FAST-NOT:  .eabi_attribute 21
; CORTEX-M4-SOFT-FAST-NOT:  .eabi_attribute 22
; CORTEX-M4-SOFT-FAST:  .eabi_attribute 23, 1

; CORTEX-M4-HARD:  .cpu cortex-m4
; CORTEX-M4-HARD:  .eabi_attribute 6, 13
; CORTEX-M4-HARD:  .eabi_attribute 7, 77
; CORTEX-M4-HARD:  .eabi_attribute 8, 0
; CORTEX-M4-HARD:  .eabi_attribute 9, 2
; CORTEX-M4-HARD:  .fpu fpv4-sp-d16
; CORTEX-M4-HARD:  .eabi_attribute 27, 1
; CORTEX-M4-HARD:  .eabi_attribute 36, 1
; CORTEX-M4-HARD-NOT:  .eabi_attribute 42
; CORTEX-M4-HARD-NOT:  .eabi_attribute 44
; CORTEX-M4-HARD-NOT:  .eabi_attribute 68
; CORTEX-M4-HARD-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-M4-HARD:  .eabi_attribute 20, 1
; CORTEX-M4-HARD:  .eabi_attribute 21, 1
; CORTEX-M4-HARD-NOT:  .eabi_attribute 22
; CORTEX-M4-HARD:  .eabi_attribute 23, 3
; CORTEX-M4-HARD:  .eabi_attribute 24, 1
; CORTEX-M4-HARD:  .eabi_attribute 25, 1
; CORTEX-M4-HARD:  .eabi_attribute 28, 1
; CORTEX-M4-HARD:  .eabi_attribute 38, 1

; CORTEX-M4-HARD-FAST-NOT:   .eabi_attribute 19
;; The M4 defaults to a VFPv4 FPU, so it flushes preserving the sign when
;; -ffast-math is specified.
; CORTEX-M4-HARD-FAST:  .eabi_attribute 20, 2
; CORTEX-M4-HARD-FAST-NOT:  .eabi_attribute 21
; CORTEX-M4-HARD-FAST-NOT:  .eabi_attribute 22
; CORTEX-M4-HARD-FAST:  .eabi_attribute 23, 1

; CORTEX-M7:  .cpu    cortex-m7
; CORTEX-M7:  .eabi_attribute 6, 13
; CORTEX-M7:  .eabi_attribute 7, 77
; CORTEX-M7:  .eabi_attribute 8, 0
; CORTEX-M7:  .eabi_attribute 9, 2
; CORTEX-M7-SOFT-NOT: .fpu
; CORTEX-M7-SINGLE:  .fpu fpv5-sp-d16
; CORTEX-M7-DOUBLE:  .fpu fpv5-d16
; CORTEX-M7-SOFT-NOT: .eabi_attribute 27
; CORTEX-M7-SINGLE:  .eabi_attribute 27, 1
; CORTEX-M7-DOUBLE-NOT: .eabi_attribute 27
; CORTEX-M7:  .eabi_attribute 36, 1
; CORTEX-M7-NOT:  .eabi_attribute 44
; CORTEX-M7:  .eabi_attribute 17, 1
; CORTEX-M7-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-M7:  .eabi_attribute 20, 1
; CORTEX-M7:  .eabi_attribute 21, 1
; CORTEX-M7-NOT:  .eabi_attribute 22
; CORTEX-M7:  .eabi_attribute 23, 3
; CORTEX-M7:  .eabi_attribute 24, 1
; CORTEX-M7:  .eabi_attribute 25, 1
; CORTEX-M7:  .eabi_attribute 38, 1
; CORTEX-M7:  .eabi_attribute 14, 0

; CORTEX-M7-NOFPU-FAST-NOT:   .eabi_attribute 19
;; The M7 has the ARMv8 FP unit, which always flushes preserving sign.
; CORTEX-M7-FAST:  .eabi_attribute 20, 2
;; Despite there being no FPU, we chose to flush to zero preserving
;; sign. This matches what the hardware would do for this architecture
;; revision.
; CORTEX-M7-NOFPU-FAST: .eabi_attribute 20, 2
; CORTEX-M7-NOFPU-FAST-NOT:  .eabi_attribute 21
; CORTEX-M7-NOFPU-FAST-NOT:  .eabi_attribute 22
; CORTEX-M7-NOFPU-FAST:  .eabi_attribute 23, 1

; CORTEX-R4:  .cpu cortex-r4
; CORTEX-R4:  .eabi_attribute 6, 10
; CORTEX-R4:  .eabi_attribute 7, 82
; CORTEX-R4:  .eabi_attribute 8, 1
; CORTEX-R4:  .eabi_attribute 9, 2
; CORTEX-R4-NOT:  .fpu vfpv3-d16
; CORTEX-R4-NOT:  .eabi_attribute 36
; CORTEX-R4-NOT:  .eabi_attribute 42
; CORTEX-R4-NOT:  .eabi_attribute 44
; CORTEX-R4-NOT:  .eabi_attribute 68
; CORTEX-R4-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-R4:  .eabi_attribute 20, 1
; CORTEX-R4:  .eabi_attribute 21, 1
; CORTEX-R4-NOT:  .eabi_attribute 22
; CORTEX-R4:  .eabi_attribute 23, 3
; CORTEX-R4:  .eabi_attribute 24, 1
; CORTEX-R4:  .eabi_attribute 25, 1
; CORTEX-R4-NOT:  .eabi_attribute 28
; CORTEX-R4:  .eabi_attribute 38, 1

; CORTEX-R4F:  .cpu cortex-r4f
; CORTEX-R4F:  .eabi_attribute 6, 10
; CORTEX-R4F:  .eabi_attribute 7, 82
; CORTEX-R4F:  .eabi_attribute 8, 1
; CORTEX-R4F:  .eabi_attribute 9, 2
; CORTEX-R4F:  .fpu vfpv3-d16
; CORTEX-R4F-NOT:  .eabi_attribute 27, 1
; CORTEX-R4F-NOT:  .eabi_attribute 36
; CORTEX-R4F-NOT:  .eabi_attribute 42
; CORTEX-R4F-NOT:  .eabi_attribute 44
; CORTEX-R4F-NOT:  .eabi_attribute 68
; CORTEX-R4F-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-R4F:  .eabi_attribute 20, 1
; CORTEX-R4F:  .eabi_attribute 21, 1
; CORTEX-R4F-NOT:  .eabi_attribute 22
; CORTEX-R4F:  .eabi_attribute 23, 3
; CORTEX-R4F:  .eabi_attribute 24, 1
; CORTEX-R4F:  .eabi_attribute 25, 1
; CORTEX-R4F-NOT:  .eabi_attribute 28
; CORTEX-R4F:  .eabi_attribute 38, 1

; CORTEX-R5:  .cpu cortex-r5
; CORTEX-R5:  .eabi_attribute 6, 10
; CORTEX-R5:  .eabi_attribute 7, 82
; CORTEX-R5:  .eabi_attribute 8, 1
; CORTEX-R5:  .eabi_attribute 9, 2
; CORTEX-R5:  .fpu vfpv3-d16
; CORTEX-R5-NOT:  .eabi_attribute 27, 1
; CORTEX-R5-NOT:  .eabi_attribute 36
; CORTEX-R5:  .eabi_attribute 44, 2
; CORTEX-R5-NOT:  .eabi_attribute 42
; CORTEX-R5-NOT:  .eabi_attribute 68
; CORTEX-R5-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-R5:  .eabi_attribute 20, 1
; CORTEX-R5:  .eabi_attribute 21, 1
; CORTEX-R5-NOT:  .eabi_attribute 22
; CORTEX-R5:  .eabi_attribute 23, 3
; CORTEX-R5:  .eabi_attribute 24, 1
; CORTEX-R5:  .eabi_attribute 25, 1
; CORTEX-R5-NOT:  .eabi_attribute 28
; CORTEX-R5:  .eabi_attribute 38, 1

; CORTEX-R5-FAST-NOT:   .eabi_attribute 19
;; The R5 has the VFPv3 FP unit, which always flushes preserving sign.
; CORTEX-R5-FAST:  .eabi_attribute 20, 2
; CORTEX-R5-FAST-NOT:  .eabi_attribute 21
; CORTEX-R5-FAST-NOT:  .eabi_attribute 22
; CORTEX-R5-FAST:  .eabi_attribute 23, 1

; CORTEX-R7:  .cpu cortex-r7
; CORTEX-R7:  .eabi_attribute 6, 10
; CORTEX-R7:  .eabi_attribute 7, 82
; CORTEX-R7:  .eabi_attribute 8, 1
; CORTEX-R7:  .eabi_attribute 9, 2
; CORTEX-R7:  .fpu vfpv3-d16-fp16
; CORTEX-R7:  .eabi_attribute 36, 1
; CORTEX-R7:  .eabi_attribute 42, 1
; CORTEX-R7:  .eabi_attribute 44, 2
; CORTEX-R7-NOT:  .eabi_attribute 68
; CORTEX-R7-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-R7:  .eabi_attribute 20, 1
; CORTEX-R7:  .eabi_attribute 21, 1
; CORTEX-R7-NOT:  .eabi_attribute 22
; CORTEX-R7:  .eabi_attribute 23, 3
; CORTEX-R7:  .eabi_attribute 24, 1
; CORTEX-R7:  .eabi_attribute 25, 1
; CORTEX-R7-NOT:  .eabi_attribute 28
; CORTEX-R7:  .eabi_attribute 38, 1

; CORTEX-R7-FAST-NOT:   .eabi_attribute 19
;; The R7 has the VFPv3 FP unit, which always flushes preserving sign.
; CORTEX-R7-FAST:  .eabi_attribute 20, 2
; CORTEX-R7-FAST-NOT:  .eabi_attribute 21
; CORTEX-R7-FAST-NOT:  .eabi_attribute 22
; CORTEX-R7-FAST:  .eabi_attribute 23, 1

; CORTEX-R8:  .cpu cortex-r8
; CORTEX-R8:  .eabi_attribute 6, 10
; CORTEX-R8:  .eabi_attribute 7, 82
; CORTEX-R8:  .eabi_attribute 8, 1
; CORTEX-R8:  .eabi_attribute 9, 2
; CORTEX-R8:  .fpu vfpv3-d16-fp16
; CORTEX-R8:  .eabi_attribute 36, 1
; CORTEX-R8:  .eabi_attribute 42, 1
; CORTEX-R8:  .eabi_attribute 44, 2
; CORTEX-R8-NOT:  .eabi_attribute 68
; CORTEX-R8-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-R8:  .eabi_attribute 20, 1
; CORTEX-R8:  .eabi_attribute 21, 1
; CORTEX-R8-NOT:  .eabi_attribute 22
; CORTEX-R8:  .eabi_attribute 23, 3
; CORTEX-R8:  .eabi_attribute 24, 1
; CORTEX-R8:  .eabi_attribute 25, 1
; CORTEX-R8-NOT:  .eabi_attribute 28
; CORTEX-R8:  .eabi_attribute 38, 1

; CORTEX-R8-FAST-NOT:   .eabi_attribute 19
;; The R8 has the VFPv3 FP unit, which always flushes preserving sign.
; CORTEX-R8-FAST:  .eabi_attribute 20, 2
; CORTEX-R8-FAST-NOT:  .eabi_attribute 21
; CORTEX-R8-FAST-NOT:  .eabi_attribute 22
; CORTEX-R8-FAST:  .eabi_attribute 23, 1

; CORTEX-A32:  .cpu cortex-a32
; CORTEX-A32:  .eabi_attribute 6, 14
; CORTEX-A32:  .eabi_attribute 7, 65
; CORTEX-A32:  .eabi_attribute 8, 1
; CORTEX-A32:  .eabi_attribute 9, 2
; CORTEX-A32:  .fpu crypto-neon-fp-armv8
; CORTEX-A32:  .eabi_attribute 12, 3
; CORTEX-A32-NOT:  .eabi_attribute 27
; CORTEX-A32:  .eabi_attribute 36, 1
; CORTEX-A32:  .eabi_attribute 42, 1
; CORTEX-A32-NOT:  .eabi_attribute 44
; CORTEX-A32:  .eabi_attribute 68, 3
; CORTEX-A32-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-A32:  .eabi_attribute 20, 1
; CORTEX-A32:  .eabi_attribute 21, 1
; CORTEX-A32-NOT:  .eabi_attribute 22
; CORTEX-A32:  .eabi_attribute 23, 3
; CORTEX-A32:  .eabi_attribute 24, 1
; CORTEX-A32:  .eabi_attribute 25, 1
; CORTEX-A32-NOT:  .eabi_attribute 28
; CORTEX-A32:  .eabi_attribute 38, 1

; CORTEX-A32-FAST-NOT:   .eabi_attribute 19
;; The A32 has the ARMv8 FP unit, which always flushes preserving sign.
; CORTEX-A32-FAST:  .eabi_attribute 20, 2
; CORTEX-A32-FAST-NOT:  .eabi_attribute 21
; CORTEX-A32-FAST-NOT:  .eabi_attribute 22
; CORTEX-A32-FAST:  .eabi_attribute 23, 1

; CORTEX-M23:  .cpu cortex-m23
; CORTEX-M23:  .eabi_attribute 6, 16
; CORTEX-M23:  .eabi_attribute 7, 77
; CORTEX-M23:  .eabi_attribute 8, 0
; CORTEX-M23:  .eabi_attribute 9, 3
; CORTEX-M23-NOT:  .eabi_attribute 27
; CORTEX-M23:  .eabi_attribute 34, 1
; CORTEX-M23-NOT:  .eabi_attribute 44
; CORTEX-M23:  .eabi_attribute 17, 1
;; We default to IEEE 754 compliance
; CORTEX-M23-NOT:   .eabi_attribute 19
; CORTEX-M23:  .eabi_attribute 20, 1
; CORTEX-M23:  .eabi_attribute 21, 1
; CORTEX-M23:  .eabi_attribute 23, 3
; CORTEX-M23:  .eabi_attribute 24, 1
; CORTEX-M23-NOT:  .eabi_attribute 28
; CORTEX-M23:  .eabi_attribute 25, 1
; CORTEX-M23:  .eabi_attribute 38, 1
; CORTEX-M23:  .eabi_attribute 14, 0

; CORTEX-M33:  .cpu cortex-m33
; CORTEX-M33:  .eabi_attribute 6, 17
; CORTEX-M33:  .eabi_attribute 7, 77
; CORTEX-M33:  .eabi_attribute 8, 0
; CORTEX-M33:  .eabi_attribute 9, 3
; CORTEX-M33:  .fpu fpv5-sp-d16
; CORTEX-M33:  .eabi_attribute 27, 1
; CORTEX-M33:  .eabi_attribute 36, 1
; CORTEX-M33-NOT:  .eabi_attribute 44
; CORTEX-M33:  .eabi_attribute 46, 1
; CORTEX-M33:  .eabi_attribute 34, 1
; CORTEX-M33:  .eabi_attribute 17, 1
;; We default to IEEE 754 compliance
; CORTEX-M23-NOT:   .eabi_attribute 19
; CORTEX-M33:  .eabi_attribute 20, 1
; CORTEX-M33:  .eabi_attribute 21, 1
; CORTEX-M33:  .eabi_attribute 23, 3
; CORTEX-M33:  .eabi_attribute 24, 1
; CORTEX-M33:  .eabi_attribute 25, 1
; CORTEX-M33-NOT:  .eabi_attribute 28
; CORTEX-M33:  .eabi_attribute 38, 1
; CORTEX-M33:  .eabi_attribute 14, 0

; CORTEX-M33-FAST-NOT:   .eabi_attribute 19
; CORTEX-M33-FAST:  .eabi_attribute 20, 2
; CORTEX-M33-FAST-NOT:  .eabi_attribute 21
; CORTEX-M33-FAST-NOT:  .eabi_attribute 22
; CORTEX-M33-FAST:  .eabi_attribute 23, 1

; CORTEX-A35:  .cpu cortex-a35
; CORTEX-A35:  .eabi_attribute 6, 14
; CORTEX-A35:  .eabi_attribute 7, 65
; CORTEX-A35:  .eabi_attribute 8, 1
; CORTEX-A35:  .eabi_attribute 9, 2
; CORTEX-A35:  .fpu crypto-neon-fp-armv8
; CORTEX-A35:  .eabi_attribute 12, 3
; CORTEX-A35-NOT:  .eabi_attribute 27
; CORTEX-A35:  .eabi_attribute 36, 1
; CORTEX-A35:  .eabi_attribute 42, 1
; CORTEX-A35-NOT:  .eabi_attribute 44
; CORTEX-A35:  .eabi_attribute 68, 3
; CORTEX-A35-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-A35:  .eabi_attribute 20, 1
; CORTEX-A35:  .eabi_attribute 21, 1
; CORTEX-A35-NOT:  .eabi_attribute 22
; CORTEX-A35:  .eabi_attribute 23, 3
; CORTEX-A35:  .eabi_attribute 24, 1
; CORTEX-A35:  .eabi_attribute 25, 1
; CORTEX-A35-NOT:  .eabi_attribute 28
; CORTEX-A35:  .eabi_attribute 38, 1

; CORTEX-A35-FAST-NOT:   .eabi_attribute 19
;; The A35 has the ARMv8 FP unit, which always flushes preserving sign.
; CORTEX-A35-FAST:  .eabi_attribute 20, 2
; CORTEX-A35-FAST-NOT:  .eabi_attribute 21
; CORTEX-A35-FAST-NOT:  .eabi_attribute 22
; CORTEX-A35-FAST:  .eabi_attribute 23, 1

; CORTEX-A53:  .cpu cortex-a53
; CORTEX-A53:  .eabi_attribute 6, 14
; CORTEX-A53:  .eabi_attribute 7, 65
; CORTEX-A53:  .eabi_attribute 8, 1
; CORTEX-A53:  .eabi_attribute 9, 2
; CORTEX-A53:  .fpu crypto-neon-fp-armv8
; CORTEX-A53:  .eabi_attribute 12, 3
; CORTEX-A53-NOT:  .eabi_attribute 27
; CORTEX-A53:  .eabi_attribute 36, 1
; CORTEX-A53:  .eabi_attribute 42, 1
; CORTEX-A53-NOT:  .eabi_attribute 44
; CORTEX-A53:  .eabi_attribute 68, 3
; CORTEX-A53-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-A53:  .eabi_attribute 20, 1
; CORTEX-A53:  .eabi_attribute 21, 1
; CORTEX-A53-NOT:  .eabi_attribute 22
; CORTEX-A53:  .eabi_attribute 23, 3
; CORTEX-A53:  .eabi_attribute 24, 1
; CORTEX-A53:  .eabi_attribute 25, 1
; CORTEX-A53-NOT:  .eabi_attribute 28
; CORTEX-A53:  .eabi_attribute 38, 1

; CORTEX-A53-FAST-NOT:   .eabi_attribute 19
;; The A53 has the ARMv8 FP unit, which always flushes preserving sign.
; CORTEX-A53-FAST:  .eabi_attribute 20, 2
; CORTEX-A53-FAST-NOT:  .eabi_attribute 21
; CORTEX-A53-FAST-NOT:  .eabi_attribute 22
; CORTEX-A53-FAST:  .eabi_attribute 23, 1

; CORTEX-A57:  .cpu cortex-a57
; CORTEX-A57:  .eabi_attribute 6, 14
; CORTEX-A57:  .eabi_attribute 7, 65
; CORTEX-A57:  .eabi_attribute 8, 1
; CORTEX-A57:  .eabi_attribute 9, 2
; CORTEX-A57:  .fpu crypto-neon-fp-armv8
; CORTEX-A57:  .eabi_attribute 12, 3
; CORTEX-A57-NOT:  .eabi_attribute 27
; CORTEX-A57:  .eabi_attribute 36, 1
; CORTEX-A57:  .eabi_attribute 42, 1
; CORTEX-A57-NOT:  .eabi_attribute 44
; CORTEX-A57:  .eabi_attribute 68, 3
; CORTEX-A57-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-A57:  .eabi_attribute 20, 1
; CORTEX-A57:  .eabi_attribute 21, 1
; CORTEX-A57-NOT:  .eabi_attribute 22
; CORTEX-A57:  .eabi_attribute 23, 3
; CORTEX-A57:  .eabi_attribute 24, 1
; CORTEX-A57:  .eabi_attribute 25, 1
; CORTEX-A57-NOT:  .eabi_attribute 28
; CORTEX-A57:  .eabi_attribute 38, 1

; CORTEX-A57-FAST-NOT:   .eabi_attribute 19
;; The A57 has the ARMv8 FP unit, which always flushes preserving sign.
; CORTEX-A57-FAST:  .eabi_attribute 20, 2
; CORTEX-A57-FAST-NOT:  .eabi_attribute 21
; CORTEX-A57-FAST-NOT:  .eabi_attribute 22
; CORTEX-A57-FAST:  .eabi_attribute 23, 1

; CORTEX-A72:  .cpu cortex-a72
; CORTEX-A72:  .eabi_attribute 6, 14
; CORTEX-A72:  .eabi_attribute 7, 65
; CORTEX-A72:  .eabi_attribute 8, 1
; CORTEX-A72:  .eabi_attribute 9, 2
; CORTEX-A72:  .fpu crypto-neon-fp-armv8
; CORTEX-A72:  .eabi_attribute 12, 3
; CORTEX-A72-NOT:  .eabi_attribute 27
; CORTEX-A72:  .eabi_attribute 36, 1
; CORTEX-A72:  .eabi_attribute 42, 1
; CORTEX-A72-NOT:  .eabi_attribute 44
; CORTEX-A72:  .eabi_attribute 68, 3
; CORTEX-A72-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-A72:  .eabi_attribute 20, 1
; CORTEX-A72:  .eabi_attribute 21, 1
; CORTEX-A72-NOT:  .eabi_attribute 22
; CORTEX-A72:  .eabi_attribute 23, 3
; CORTEX-A72:  .eabi_attribute 24, 1
; CORTEX-A72:  .eabi_attribute 25, 1
; CORTEX-A72-NOT:  .eabi_attribute 28
; CORTEX-A72:  .eabi_attribute 38, 1

; CORTEX-A72-FAST-NOT:   .eabi_attribute 19
;; The A72 has the ARMv8 FP unit, which always flushes preserving sign.
; CORTEX-A72-FAST:  .eabi_attribute 20, 2
; CORTEX-A72-FAST-NOT:  .eabi_attribute 21
; CORTEX-A72-FAST-NOT:  .eabi_attribute 22
; CORTEX-A72-FAST:  .eabi_attribute 23, 1

; CORTEX-A73:  .cpu cortex-a73
; CORTEX-A73:  .eabi_attribute 6, 14
; CORTEX-A73:  .eabi_attribute 7, 65
; CORTEX-A73:  .eabi_attribute 8, 1
; CORTEX-A73:  .eabi_attribute 9, 2
; CORTEX-A73:  .fpu  crypto-neon-fp-armv8
; CORTEX-A73:  .eabi_attribute 12, 3
; CORTEX-A73-NOT: .eabi_attribute 27
; CORTEX-A73:  .eabi_attribute 36, 1
; CORTEX-A73:  .eabi_attribute 42, 1
; CORTEX-A73-NOT: .eabi_attribute 44
; CORTEX-A73:  .eabi_attribute 68, 3
; CORTEX-A73-NOT: .eabi_attribute 19
;; We default to IEEE 754 compliance
; CORTEX-A73:  .eabi_attribute 20, 1
; CORTEX-A73:  .eabi_attribute 21, 1
; CORTEX-A73-NOT:  .eabi_attribute 22
; CORTEX-A73:  .eabi_attribute 23, 3
; CORTEX-A73:  .eabi_attribute 24, 1
; CORTEX-A73:  .eabi_attribute 25, 1
; CORTEX-A73-NOT: .eabi_attribute 28
; CORTEX-A73:  .eabi_attribute 38, 1
; CORTEX-A73:  .eabi_attribute 14, 0

; EXYNOS-M1:  .cpu exynos-m1
; EXYNOS-M1:  .eabi_attribute 6, 14
; EXYNOS-M1:  .eabi_attribute 7, 65
; EXYNOS-M1:  .eabi_attribute 8, 1
; EXYNOS-M1:  .eabi_attribute 9, 2
; EXYNOS-M1:  .fpu crypto-neon-fp-armv8
; EXYNOS-M1:  .eabi_attribute 12, 3
; EXYNOS-M1-NOT:  .eabi_attribute 27
; EXYNOS-M1:  .eabi_attribute 36, 1
; EXYNOS-M1:  .eabi_attribute 42, 1
; EXYNOS-M1-NOT:  .eabi_attribute 44
; EXYNOS-M1:  .eabi_attribute 68, 3
; EXYNOS-M1-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; EXYNOS-M1:  .eabi_attribute 20, 1
; EXYNOS-M1:  .eabi_attribute 21, 1
; EXYNOS-M1-NOT:  .eabi_attribute 22
; EXYNOS-M1:  .eabi_attribute 23, 3
; EXYNOS-M1:  .eabi_attribute 24, 1
; EXYNOS-M1:  .eabi_attribute 25, 1
; EXYNOS-M1-NOT:  .eabi_attribute 28
; EXYNOS-M1:  .eabi_attribute 38, 1

; EXYNOS-M1-FAST-NOT:   .eabi_attribute 19
;; The exynos-m1 has the ARMv8 FP unit, which always flushes preserving sign.
; EXYNOS-M1-FAST:  .eabi_attribute 20, 2
; EXYNOS-M1-FAST-NOT:  .eabi_attribute 21
; EXYNOS-M1-FAST-NOT:  .eabi_attribute 22
; EXYNOS-M1-FAST:  .eabi_attribute 23, 1

; EXYNOS-M2:  .cpu exynos-m2
; EXYNOS-M2:  .eabi_attribute 6, 14
; EXYNOS-M2:  .eabi_attribute 7, 65
; EXYNOS-M2:  .eabi_attribute 8, 1
; EXYNOS-M2:  .eabi_attribute 9, 2
; EXYNOS-M2:  .fpu crypto-neon-fp-armv8
; EXYNOS-M2:  .eabi_attribute 12, 3
; EXYNOS-M2-NOT:  .eabi_attribute 27
; EXYNOS-M2:  .eabi_attribute 36, 1
; EXYNOS-M2:  .eabi_attribute 42, 1
; EXYNOS-M2-NOT:  .eabi_attribute 44
; EXYNOS-M2:  .eabi_attribute 68, 3
; EXYNOS-M2-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; EXYNOS-M2:  .eabi_attribute 20, 1
; EXYNOS-M2:  .eabi_attribute 21, 1
; EXYNOS-M2-NOT:  .eabi_attribute 22
; EXYNOS-M2:  .eabi_attribute 23, 3
; EXYNOS-M2:  .eabi_attribute 24, 1
; EXYNOS-M2:  .eabi_attribute 25, 1
; EXYNOS-M2-NOT:  .eabi_attribute 28
; EXYNOS-M2:  .eabi_attribute 38, 1

; EXYNOS-M3:  .cpu exynos-m3
; EXYNOS-M3:  .eabi_attribute 6, 14
; EXYNOS-M3:  .eabi_attribute 7, 65
; EXYNOS-M3:  .eabi_attribute 8, 1
; EXYNOS-M3:  .eabi_attribute 9, 2
; EXYNOS-M3:  .fpu crypto-neon-fp-armv8
; EXYNOS-M3:  .eabi_attribute 12, 3
; EXYNOS-M3-NOT:  .eabi_attribute 27
; EXYNOS-M3:  .eabi_attribute 36, 1
; EXYNOS-M3:  .eabi_attribute 42, 1
; EXYNOS-M3-NOT:  .eabi_attribute 44
; EXYNOS-M3:  .eabi_attribute 68, 3
; EXYNOS-M3-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; EXYNOS-M3:  .eabi_attribute 20, 1
; EXYNOS-M3:  .eabi_attribute 21, 1
; EXYNOS-M3-NOT:  .eabi_attribute 22
; EXYNOS-M3:  .eabi_attribute 23, 3
; EXYNOS-M3:  .eabi_attribute 24, 1
; EXYNOS-M3:  .eabi_attribute 25, 1
; EXYNOS-M3-NOT:  .eabi_attribute 28
; EXYNOS-M3:  .eabi_attribute 38, 1

; GENERIC-FPU-VFPV3-FP16: .fpu vfpv3-fp16
; GENERIC-FPU-VFPV3-D16-FP16: .fpu vfpv3-d16-fp16
; GENERIC-FPU-VFPV3XD: .fpu vfpv3xd
; GENERIC-FPU-VFPV3XD-FP16: .fpu vfpv3xd-fp16
; GENERIC-FPU-NEON-FP16: .fpu neon-fp16

; GENERIC-ARMV8_1-A:  .eabi_attribute 6, 14
; GENERIC-ARMV8_1-A:  .eabi_attribute 7, 65
; GENERIC-ARMV8_1-A:  .eabi_attribute 8, 1
; GENERIC-ARMV8_1-A:  .eabi_attribute 9, 2
; GENERIC-ARMV8_1-A:  .fpu crypto-neon-fp-armv8
; GENERIC-ARMV8_1-A:  .eabi_attribute 12, 4
; GENERIC-ARMV8_1-A-NOT:  .eabi_attribute 27
; GENERIC-ARMV8_1-A:  .eabi_attribute 36, 1
; GENERIC-ARMV8_1-A:  .eabi_attribute 42, 1
; GENERIC-ARMV8_1-A-NOT:  .eabi_attribute 44
; GENERIC-ARMV8_1-A:  .eabi_attribute 68, 3
; GENERIC-ARMV8_1-A-NOT:   .eabi_attribute 19
;; We default to IEEE 754 compliance
; GENERIC-ARMV8_1-A:  .eabi_attribute 20, 1
; GENERIC-ARMV8_1-A:  .eabi_attribute 21, 1
; GENERIC-ARMV8_1-A-NOT:  .eabi_attribute 22
; GENERIC-ARMV8_1-A:  .eabi_attribute 23, 3
; GENERIC-ARMV8_1-A:  .eabi_attribute 24, 1
; GENERIC-ARMV8_1-A:  .eabi_attribute 25, 1
; GENERIC-ARMV8_1-A-NOT:  .eabi_attribute 28
; GENERIC-ARMV8_1-A:  .eabi_attribute 38, 1

; GENERIC-ARMV8_1-A-FAST-NOT:   .eabi_attribute 19
;; GENERIC-ARMV8_1-A has the ARMv8 FP unit, which always flushes preserving sign.
; GENERIC-ARMV8_1-A-FAST:  .eabi_attribute 20, 2
; GENERIC-ARMV8_1-A-FAST-NOT:  .eabi_attribute 21
; GENERIC-ARMV8_1-A-FAST-NOT:  .eabi_attribute 22
; GENERIC-ARMV8_1-A-FAST:  .eabi_attribute 23, 1

; RELOC-PIC:  .eabi_attribute 15, 1
; RELOC-PIC:  .eabi_attribute 16, 1
; RELOC-PIC:  .eabi_attribute 17, 2
; RELOC-OTHER:  .eabi_attribute 17, 1
; RELOC-ROPI-NOT:  .eabi_attribute 15,
; RELOC-ROPI:      .eabi_attribute 16, 1
; RELOC-ROPI:      .eabi_attribute 17, 1
; RELOC-RWPI:      .eabi_attribute 15, 2
; RELOC-RWPI-NOT:  .eabi_attribute 16,
; RELOC-RWPI:      .eabi_attribute 17, 1
; RELOC-ROPI-RWPI: .eabi_attribute 15, 2
; RELOC-ROPI-RWPI: .eabi_attribute 16, 1
; RELOC-ROPI-RWPI:  .eabi_attribute 17, 1

; PCS-R9-USE:  .eabi_attribute 14, 0
; PCS-R9-RESERVE:  .eabi_attribute 14, 3

; ARMv8R: .eabi_attribute 67, "2.09"      @ Tag_conformance
; ARMv8R: .eabi_attribute 6, 15   @ Tag_CPU_arch
; ARMv8R: .eabi_attribute 7, 82   @ Tag_CPU_arch_profile
; ARMv8R: .eabi_attribute 8, 1    @ Tag_ARM_ISA_use
; ARMv8R: .eabi_attribute 9, 2    @ Tag_THUMB_ISA_use
; ARMv8R-NOFPU-NOT: .fpu
; ARMv8R-NOFPU-NOT: .eabi_attribute 12
; ARMv8R-SP: .fpu fpv5-sp-d16
; ARMv8R-SP-NOT: .eabi_attribute 12
; ARMv8R-NEON: .fpu    neon-fp-armv8
; ARMv8R-NEON: .eabi_attribute 12, 3   @ Tag_Advanced_SIMD_arch
; ARMv8R-NOFPU-NOT: .eabi_attribute 27
; ARMv8R-SP: .eabi_attribute 27, 1   @ Tag_ABI_HardFP_use
; ARMv8R-NEON-NOT: .eabi_attribute 27
; ARMv8R-NOFPU-NOT: .eabi_attribute 36
; ARMv8R-SP: .eabi_attribute 36, 1   @ Tag_FP_HP_extension
; ARMv8R-NEON: .eabi_attribute 36, 1   @ Tag_FP_HP_extension
; ARMv8R: .eabi_attribute 42, 1   @ Tag_MPextension_use
; ARMv8R: .eabi_attribute 68, 2   @ Tag_Virtualization_use
; ARMv8R: .eabi_attribute 38, 1   @ Tag_ABI_FP_16bit_format
; ARMv8R: .eabi_attribute 14, 0   @ Tag_ABI_PCS_R9_use

define i32 @f(i64 %z) {
    ret i32 0
}
