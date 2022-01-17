# RUN: not llvm-mc -triple=riscv32 --mattr=+experimental-v \
# RUN:             --mattr=+experimental-zvlsseg %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

vluxei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vluxei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vloxei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vloxei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsuxei64.v v24, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsuxei64.v v24, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsoxei64.v v24, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsoxei64.v v24, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vluxseg2ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vluxseg2ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vluxseg3ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vluxseg3ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vluxseg4ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vluxseg4ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vluxseg5ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vluxseg5ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vluxseg6ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vluxseg6ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vluxseg7ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vluxseg7ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vluxseg8ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vluxseg8ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vloxseg2ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vloxseg2ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vloxseg3ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vloxseg3ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vloxseg4ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vloxseg4ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vloxseg5ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vloxseg5ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vloxseg6ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vloxseg6ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vloxseg7ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vloxseg7ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vloxseg8ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vloxseg8ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsuxseg2ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsuxseg2ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsuxseg3ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsuxseg3ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsuxseg4ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsuxseg4ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsuxseg5ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsuxseg5ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsuxseg6ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsuxseg6ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsuxseg7ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsuxseg7ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsuxseg8ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsuxseg8ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsoxseg2ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsoxseg2ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsoxseg3ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsoxseg3ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsoxseg4ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsoxseg4ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsoxseg5ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsoxseg5ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsoxseg6ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsoxseg6ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsoxseg7ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsoxseg7ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsoxseg8ei64.v v8, (a0), v4, v0.t
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set

vsoxseg8ei64.v v8, (a0), v4
# CHECK-ERROR: instruction requires the following: RV64I Base Instruction Set
