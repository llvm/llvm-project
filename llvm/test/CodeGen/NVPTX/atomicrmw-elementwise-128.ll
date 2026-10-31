; RUN: llc -mtriple=nvptx64 -mcpu=sm_70 -mattr=+ptx83 -verify-machineinstrs %s -o - | FileCheck %s --check-prefixes=CHECK,SPLIT
; RUN: llc -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx83 -verify-machineinstrs %s -o - | FileCheck %s --check-prefixes=CHECK,SPLIT
; RUN: llc -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx83 -verify-machineinstrs %s -o - | FileCheck %s --check-prefixes=CHECK,WIDE
; RUN: %if ptxas-sm_80 && ptxas-isa-8.3 %{ llc -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx83 %s -o - | %ptxas-verify -arch=sm_80 %}
; RUN: %if ptxas-sm_90 && ptxas-isa-8.3 %{ llc -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx83 %s -o - | %ptxas-verify -arch=sm_90 %}

; SM 70/80 support 128-bit atomic loads/stores, but not 128-bit RMWs.
; Elementwise RMWs must split even when the whole vector fits an atomic load.

; CHECK-LABEL: exchange128(
; SPLIT-COUNT-2: atom.relaxed.gpu.global.exch.b64
; WIDE: atom.relaxed.gpu.global.exch.b128
; CHECK: ret;
define <2 x i64> @exchange128(ptr addrspace(1) %p, <2 x i64> %v) {
  %r = atomicrmw elementwise xchg ptr addrspace(1) %p, <2 x i64> %v syncscope("device") monotonic, align 16
  ret <2 x i64> %r
}

; CHECK-LABEL: exchange128_align8(
; CHECK-COUNT-2: atom.relaxed.gpu.global.exch.b64
; CHECK-NOT: atom.{{.*}}.b128
; CHECK: ret;
define <2 x i64> @exchange128_align8(ptr addrspace(1) %p, <2 x i64> %v) {
  %r = atomicrmw elementwise xchg ptr addrspace(1) %p, <2 x i64> %v syncscope("device") monotonic, align 8
  ret <2 x i64> %r
}

; Apply the same RMW width limit when choosing a vector-wide CAS loop.
; CHECK-LABEL: minimum128(
; SPLIT-COUNT-2: atom.relaxed.gpu.global.cas.b64
; WIDE: atom.relaxed.gpu.global.cas.b128
; CHECK: ret;
define <2 x double> @minimum128(ptr addrspace(1) %p, <2 x double> %v) {
  %r = atomicrmw elementwise fmin ptr addrspace(1) %p, <2 x double> %v syncscope("device") monotonic, align 16
  ret <2 x double> %r
}
