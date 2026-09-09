; RUN: split-file %s %t
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_50 -mattr=+ptx40 -filetype=null %t/cmpxchg.ll 2>&1 | FileCheck %s
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_50 -mattr=+ptx40 -filetype=null %t/cmpxchg-partword.ll 2>&1 | FileCheck %s
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_50 -mattr=+ptx40 -filetype=null %t/atomicrmw.ll 2>&1 | FileCheck %s
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_50 -mattr=+ptx40 -filetype=null %t/atomicrmw-expand.ll 2>&1 | FileCheck %s
; RUN: llc -mtriple=nvptx64 -mcpu=sm_50 -mattr=+ptx40 %t/device.ll -o - | FileCheck %s --check-prefix=DEVICE

; CHECK: LLVM ERROR: NVPTX system scope atomics require sm_60 or later

;--- cmpxchg.ll
define i32 @cmpxchg_system(ptr %addr, i32 %cmp, i32 %new) {
  %result = cmpxchg ptr %addr, i32 %cmp, i32 %new monotonic monotonic
  %value = extractvalue { i32, i1 } %result, 0
  ret i32 %value
}

;--- cmpxchg-partword.ll
define i8 @cmpxchg_system_i8(ptr %addr, i8 %cmp, i8 %new) {
  %result = cmpxchg ptr %addr, i8 %cmp, i8 %new monotonic monotonic
  %value = extractvalue { i8, i1 } %result, 0
  ret i8 %value
}

;--- atomicrmw.ll
define i32 @atomicrmw_system(ptr %addr, i32 %value) {
  %result = atomicrmw add ptr %addr, i32 %value monotonic
  ret i32 %result
}

;--- atomicrmw-expand.ll
define i32 @atomicrmw_system_nand(ptr %addr, i32 %value) {
  %result = atomicrmw nand ptr %addr, i32 %value monotonic
  ret i32 %result
}

;--- device.ll
; DEVICE-LABEL: cmpxchg_device(
; DEVICE: atom.cas.b32
define i32 @cmpxchg_device(ptr %addr, i32 %cmp, i32 %new) {
  %result = cmpxchg ptr %addr, i32 %cmp, i32 %new syncscope("device") monotonic monotonic
  %value = extractvalue { i32, i1 } %result, 0
  ret i32 %value
}

; DEVICE-LABEL: atomicrmw_device(
; DEVICE: atom.add.u32
define i32 @atomicrmw_device(ptr %addr, i32 %value) {
  %result = atomicrmw add ptr %addr, i32 %value syncscope("device") monotonic
  ret i32 %result
}
