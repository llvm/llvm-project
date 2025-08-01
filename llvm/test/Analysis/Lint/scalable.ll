; RUN: opt -S -passes=lint -disable-output < %s 2>&1 | FileCheck %s --allow-empty

; CHECK-NOT: Buffer overflow
define <vscale x 8 x i8> @alloca_access() {
  %a = alloca <vscale x 8 x i8>
  %v = load <vscale x 8 x i8>, ptr %a
  ret <vscale x 8 x i8> %v
}

; CHECK-NOT: Buffer overflow
define <vscale x 8 x i8> @alloca_access2() {
  %a = alloca <256 x i8>
  %v = load <vscale x 8 x i8>, ptr %a
  ret <vscale x 8 x i8> %v
}

; CHECK-NOT: insertelement index out of range
define <vscale x 8 x half> @insertelement() {
  %insert = insertelement <vscale x 8 x half> poison, half 0xH0000, i64 100
  ret <vscale x 8 x half> %insert
}

; CHECK-NOT: extract index out of range
define half @extractelement(<vscale x 8 x half> %v) {
  %insert = extractelement <vscale x 8 x half> %v, i64 100
  ret half %insert
}
