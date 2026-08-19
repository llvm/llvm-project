; RUN: not llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx83 -filetype=null 2>&1 | FileCheck %s

define void @bitwise_i256(ptr %0, i256 %1) {
  ; CHECK:      error: unsupported atomic load: target supports atomics up to 16 bytes, but this atomic accesses 32 bytes
  ; CHECK:      error: unsupported atomicrmw xchg: target supports atomics up to 16 bytes, but this atomic accesses 32 bytes
  ; CHECK:      error: unsupported atomic load: target supports atomics up to 16 bytes, but this atomic accesses 32 bytes
  ; CHECK:      error: unsupported atomicrmw xor: target supports atomics up to 16 bytes, but this atomic accesses 32 bytes
  ; CHECK:      error: unsupported atomic load: target supports atomics up to 16 bytes, but this atomic accesses 32 bytes
  ; CHECK:      error: unsupported atomicrmw or: target supports atomics up to 16 bytes, but this atomic accesses 32 bytes
  ; CHECK:      error: unsupported atomic load: target supports atomics up to 16 bytes, but this atomic accesses 32 bytes
  ; CHECK:      error: unsupported atomicrmw and: target supports atomics up to 16 bytes, but this atomic accesses 32 bytes
entry:
  %2 = atomicrmw and ptr %0, i256 %1 monotonic
  %3 = atomicrmw or ptr %0, i256 %1 monotonic
  %4 = atomicrmw xor ptr %0, i256 %1 monotonic
  %5 = atomicrmw xchg ptr %0, i256 %1 monotonic
  ret void
}

define void @minmax_i256(ptr %0, i256 %1) {
  ; CHECK:      error: unsupported atomic load: target supports atomics up to 16 bytes, but this atomic accesses 32 bytes
  ; CHECK:      error: unsupported atomicrmw umax: target supports atomics up to 16 bytes, but this atomic accesses 32 bytes
  ; CHECK:      error: unsupported atomic load: target supports atomics up to 16 bytes, but this atomic accesses 32 bytes
  ; CHECK:      error: unsupported atomicrmw umin: target supports atomics up to 16 bytes, but this atomic accesses 32 bytes
  ; CHECK:      error: unsupported atomic load: target supports atomics up to 16 bytes, but this atomic accesses 32 bytes
  ; CHECK:      error: unsupported atomicrmw max: target supports atomics up to 16 bytes, but this atomic accesses 32 bytes
  ; CHECK:      error: unsupported atomic load: target supports atomics up to 16 bytes, but this atomic accesses 32 bytes
  ; CHECK:      error: unsupported atomicrmw min: target supports atomics up to 16 bytes, but this atomic accesses 32 bytes
entry:
  %2 = atomicrmw min ptr %0, i256 %1 monotonic
  %3 = atomicrmw max ptr %0, i256 %1 monotonic
  %4 = atomicrmw umin ptr %0, i256 %1 monotonic
  %5 = atomicrmw umax ptr %0, i256 %1 monotonic
  ret void
}
