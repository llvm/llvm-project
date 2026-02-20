; RUN: llc --mtriple=hexagon -mattr=+hvxv79,+hvx-length128b < %s | FileCheck %s

; Test truncation of <64 x i32> (vector pair) to <64 x i1>
define void @f5(<64 x i32> %a0, ptr %a1) {
; CHECK-LABEL: f5:
; CHECK: [[REG0:(r[0-9]+)]] = ##16843009
; CHECK-DAG: q[[Q0:[0-9]+]] = vand(v{{[0-9]+}},[[REG0]])
; CHECK-DAG: q[[Q1:[0-9]+]] = vand(v{{[0-9]+}},[[REG0]])
; CHECK: v{{[0-9]+}}.b = vpacke(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
; CHECK: v{{[0-9]+}}.b = vpacke(v{{[0-9]+}}.h,v{{[0-9]+}}.h)
; CHECK: v[[VROR:[0-9]+]] = vror(v{{[0-9]+}},r{{[0-9]+}})
; CHECK: v[[VOR:[0-9]+]] = vor(v[[VROR]],v{{[0-9]+}})
; CHECK: q{{[0-9]+}} = vand(v[[VOR]],r{{[0-9]+}})
b0:
  %v0 = trunc <64 x i32> %a0 to <64 x i1>
  store <64 x i1> %v0, ptr %a1, align 1
  ret void
}

; Test truncation of <64 x i8> (half HVX width, needs widening) to <64 x i1>
; This was crashing with "Unhandled HVX operation" because the truncate
; to i1 vector case was not handled when the input vector needed widening.
define i64 @trunc_v64i8_to_v64i1(<64 x i8> %v) {
; CHECK-LABEL: trunc_v64i8_to_v64i1:
; CHECK: q{{[0-9]+}} = vand(v{{[0-9]+}},r{{[0-9]+}})
  %1 = trunc <64 x i8> %v to <64 x i1>
  %2 = bitcast <64 x i1> %1 to i64
  ret i64 %2
}

; Test truncation of <64 x i8> to <64 x i1> with store
define void @trunc_v64i8_to_v64i1_store(<64 x i8> %v, ptr %p) {
; CHECK-LABEL: trunc_v64i8_to_v64i1_store:
; CHECK: q{{[0-9]+}} = vand(v{{[0-9]+}},r{{[0-9]+}})
  %1 = trunc <64 x i8> %v to <64 x i1>
  store <64 x i1> %1, ptr %p, align 8
  ret void
}
