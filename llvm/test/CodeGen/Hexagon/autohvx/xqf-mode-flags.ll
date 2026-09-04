; Test that qfloat mode flags invoke correct code generation.

; REQUIRES: asserts
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -march=hexagon \
; RUN: -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv79 -verify-machineinstrs \
; RUN: -hexagon-qfloat-mode=strict-ieee  \
; RUN: -debug-only=hexagon-xqf-gen,hexagon-qfp-optimizer -enable-xqf-gen < %s\
; RUN:  2>&1 | FileCheck %s  --check-prefix=STRICT-IEEE
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv81 -march=hexagon \
; RUN: -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv81 -verify-machineinstrs \
; RUN: -hexagon-qfloat-mode=strict-ieee  \
; RUN: -debug-only=hexagon-xqf-gen,hexagon-qfp-optimizer -enable-xqf-gen < %s\
; RUN:  2>&1 | FileCheck %s  --check-prefix=STRICT-IEEE
; STRICT-IEEE: Generating code for STRICT-IEEE mode
; STRICT-IEEE-NOT: Running QFPOptimzer Pass


; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -march=hexagon \
; RUN: -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv79 -verify-machineinstrs \
; RUN: -hexagon-qfloat-mode=ieee \
; RUN: -debug-only=hexagon-xqf-gen,hexagon-qfp-optimizer -enable-xqf-gen < %s\
; RUN: 2>&1 | FileCheck %s  --check-prefix=IEEE
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv81 -march=hexagon \
; RUN: -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv81 -verify-machineinstrs \
; RUN: -hexagon-qfloat-mode=ieee \
; RUN: -debug-only=hexagon-xqf-gen,hexagon-qfp-optimizer -enable-xqf-gen < %s\
; RUN: 2>&1 | FileCheck %s  --check-prefix=IEEE
; IEEE: Generating code for IEEE mode
; IEEE-NOT: Running QFPOptimzer Pass


; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -march=hexagon \
; RUN: -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv79 -hexagon-qfloat-mode=lossy \
; RUN: -verify-machineinstrs -debug-only=hexagon-xqf-gen,hexagon-qfp-optimizer\
; RUN: -enable-xqf-gen  < %s 2>&1 | FileCheck %s  --check-prefix=LOSSY
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv81 -march=hexagon \
; RUN: -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv81 -hexagon-qfloat-mode=lossy \
; RUN: -verify-machineinstrs -debug-only=hexagon-xqf-gen,hexagon-qfp-optimizer\
; RUN: -enable-xqf-gen  < %s 2>&1 | FileCheck %s  --check-prefix=LOSSY
; LOSSY: Generating code for LOSSY mode
; LOSSY-NOT: Running QFPOptimzer Pass


; The default mode is LEGACY.
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -march=hexagon \
; RUN: -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv79  -verify-machineinstrs \
; RUN: -debug-only=hexagon-xqf-gen,hexagon-qfp-optimizer -enable-xqf-gen < %s\
; RUN:  2>&1 | FileCheck %s  --check-prefix=LEGACY
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv81 -march=hexagon \
; RUN: -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv81  -verify-machineinstrs \
; RUN: -debug-only=hexagon-xqf-gen,hexagon-qfp-optimizer -enable-xqf-gen < %s\
; RUN:  2>&1 | FileCheck %s  --check-prefix=LEGACY


; Test that QFloat mode pass is not invoked. Instead we should run
; the QFPOptimizer pass.
; LEGACY-NOT: Generating code for LEGACY mode
; LEGACY: Running QFPOptimzer Pass
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -march=hexagon \
; RUN: -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv79 -hexagon-qfloat-mode=legacy\
; RUN: -verify-machineinstrs -debug-only=hexagon-xqf-gen,hexagon-qfp-optimizer\
; RUN: -enable-xqf-gen  < %s 2>&1 | FileCheck %s  --check-prefix=LEGACY
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv75 -march=hexagon\
; RUN: -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv75 -hexagon-qfloat-mode=legacy\
; RUN: -verify-machineinstrs -debug-only=hexagon-xqf-gen,hexagon-qfp-optimizer\
; RUN: -enable-xqf-gen  < %s 2>&1 | FileCheck %s  --check-prefix=LEGACY
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv81 -march=hexagon\
; RUN: -mattr=+hvx-ieee-fp,+hvx-length128b,+hvxv81 -hexagon-qfloat-mode=legacy\
; RUN: -verify-machineinstrs -debug-only=hexagon-xqf-gen,hexagon-qfp-optimizer\
; RUN: -enable-xqf-gen  < %s 2>&1 | FileCheck %s  --check-prefix=LEGACY

define <64 x half> @add_qf16(<64 x half> %a0, <64 x half> %a1) #0 {
label0:
  %v0 = fadd <64 x half> %a0, %a1
  ret <64 x half> %v0
}
