; RUN: llc -mtriple=arm64-apple-ios %s -o - -mcpu=cyclone -asm-print-latency=1 | FileCheck %s --match-full-lines --check-prefix=ON
; RUN: llc -mtriple=arm64-apple-ios %s -o - -mcpu=cyclone -asm-print-latency=0 | FileCheck %s --match-full-lines --check-prefix=OFF
; RUN: llc -mtriple=arm64-apple-ios %s -o - -mcpu=cyclone                      | FileCheck %s --match-full-lines --check-prefix=OFF

define <4 x i64> @load_v4i64(ptr %ptr){
; ON:     ldp q0, q1, [x0] ; Latency: 4
; OFF:    ldp q0, q1, [x0]
  %a = load <4 x i64>, ptr %ptr
  ret <4 x i64> %a
}
