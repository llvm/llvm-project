; check if Post RA pass handles qf types in all xqf modes.
; Since legacy mode is the default mode, check if the pass
; runs without hexagon-qfloat-mode flag explicitly set.

; REQUIRES: asserts
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -force-hvx-float -enable-xqf-gen=true \
; RUN: -hexagon-qfloat-mode=legacy -mattr=+hvxv79,+hvx-length128B -debug-only=handle-qfp \
; RUN: 2>&1 < %s -o /dev/null | FileCheck %s --check-prefix LEGACY
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -force-hvx-float -enable-xqf-gen=true \
; RUN: -mattr=+hvxv79,+hvx-length128B -debug-only=handle-qfp \
; RUN: 2>&1 < %s -o /dev/null | FileCheck %s --check-prefix LEGACY
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -force-hvx-float -enable-xqf-gen=true \
; RUN: -hexagon-qfloat-mode=lossy -mattr=+hvxv79,+hvx-length128B -debug-only=handle-qfp \
; RUN: 2>&1 < %s -o /dev/null | FileCheck %s --check-prefix LOSSY
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -force-hvx-float -enable-xqf-gen=true \
; RUN: -hexagon-qfloat-mode=ieee -mattr=+hvxv79,+hvx-length128B -debug-only=handle-qfp \
; RUN: 2>&1 < %s -o /dev/null | FileCheck %s --check-prefix IEEE
; RUN: llc -O2 -march=hexagon -mcpu=hexagonv79 -force-hvx-float -enable-xqf-gen=true \
; RUN: -hexagon-qfloat-mode=strict-ieee -mattr=+hvxv79,+hvx-length128B -debug-only=handle-qfp \
; RUN: 2>&1 < %s -o /dev/null | FileCheck %s --check-prefix STRICT-IEEE

; LEGACY: Mode: Legacy
; LOSSY: Mode: Lossy
; IEEE: Mode: IEEE
; STRICT-IEEE: Mode: Strict IEEE

define i32 @foo() {
entry:
  ret i32 0
}
