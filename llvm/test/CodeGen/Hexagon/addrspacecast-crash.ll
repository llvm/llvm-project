; Tests if addrspacecast is handled in Hexagon backend

; REQUIRES: asserts

; RUN: llc -march=hexagon %s -o /dev/null

define double @f(ptr %G, ptr %x) {
BB:
  %Castaddrspacecast = addrspacecast ptr %x to ptr addrspace(1)
  store ptr addrspace(1) %Castaddrspacecast, ptr %G, align 8
  ret double 0.000000e+00
}
