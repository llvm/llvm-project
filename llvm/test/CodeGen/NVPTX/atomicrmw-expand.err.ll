; RUN: not llc -mtriple=nvptx64 -mcpu=sm_30 -filetype=null %s 2>&1 | FileCheck %s

; CHECK: error: unsupported cmpxchg
; CHECK: error: unsupported cmpxchg
; CHECK: error: unsupported cmpxchg
; CHECK: error: unsupported cmpxchg
define void @bitwise_i256(ptr %0, i256 %1) {
entry:
  %2 = atomicrmw and ptr %0, i256 %1 monotonic, align 16
  %3 = atomicrmw or ptr %0, i256 %1 monotonic, align 16
  %4 = atomicrmw xor ptr %0, i256 %1 monotonic, align 16
  %5 = atomicrmw xchg ptr %0, i256 %1 monotonic, align 16
  ret void
}

; CHECK: error: unsupported cmpxchg
; CHECK: error: unsupported cmpxchg
; CHECK: error: unsupported cmpxchg
; CHECK: error: unsupported cmpxchg
define void @minmax_i256(ptr %0, i256 %1) {
entry:
  %2 = atomicrmw min ptr %0, i256 %1 monotonic, align 16
  %3 = atomicrmw max ptr %0, i256 %1 monotonic, align 16
  %4 = atomicrmw umin ptr %0, i256 %1 monotonic, align 16
  %5 = atomicrmw umax ptr %0, i256 %1 monotonic, align 16
  ret void
}
