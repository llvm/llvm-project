; RUN: llc -march=hexagon < %s
; REQUIRES: asserts

; Only simple types memory accesses are handled.

target triple = "hexagon"

%struct.hoge = type { i320 }

define dso_local void @widget() {
bb:
  %tmp = alloca %struct.hoge, align 1
  %tmp1 = bitcast %struct.hoge* %tmp to i320*
  %tmp2 = load i320, i320* %tmp1, align 1
  %tmp3 = and i320 %tmp2, -18446744073709551616
  %tmp4 = or i320 %tmp3, 0
  store i320 %tmp4, i320* %tmp1, align 1
  call void @llvm.trap()
  unreachable
}

declare void @llvm.trap()
