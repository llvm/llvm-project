; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: nofpclass must have exactly one entry
; CHECK-NEXT: !0 = !{}
define float @md_missing_value(ptr %ptr) {
  %load = load float, ptr %ptr, align 4, !nofpclass !{}
  ret float %load
}

; CHECK: nofpclass must have exactly one entry
; CHECK-NEXT: !1 = !{i32 1, i32 2}
define float @md_too_many_entries(ptr %ptr) {
  %load = load float, ptr %ptr, align 4, !nofpclass !{i32 1, i32 2}
  ret float %load
}

; CHECK: nofpclass entry must be a constant i32
; CHECK: !2 = !{i64 1}
define float @md_is_i64(ptr %ptr) {
  %load = load float, ptr %ptr, align 4, !nofpclass !{i64 1}
  ret float %load
}

; CHECK: nofpclass entry must be a constant i32
; CHECK-NEXT: !3 = !{float 1.000000e+00}
define float @md_is_float(ptr %ptr) {
  %load = load float, ptr %ptr, align 4, !nofpclass !{float 1.0}
  ret float %load
}

; CHECK: nofpclass entry must be a constant i32
; CHECK-NEXT: !4 = !{!"foo"}
define float @md_is_string(ptr %ptr) {
  %load = load float, ptr %ptr, align 4, !nofpclass !{!"foo"}
  ret float %load
}

; CHECK: nofpclass entry must be a constant i32
; CHECK-NEXT: !5 = !{ptr @md_is_ptr}
define float @md_is_ptr(ptr %ptr) {
  %load = load float, ptr %ptr, align 4, !nofpclass !{ptr @md_is_ptr}
  ret float %load
}

; CHECK: 'nofpclass' must have at least one test bit set
; CHECK-NEXT: !6 = !{i32 0}
; CHECK-NEXT: %load = load float, ptr %ptr, align 4, !nofpclass !6
  define float @md_is_zero_val(ptr %ptr) {
  %load = load float, ptr %ptr, align 4, !nofpclass !{i32 0}
  ret float %load
}

; CHECK: Invalid value for 'nofpclass' test mask
; CHECK-NEXT: !7 = !{i32 1024}
; CHECK-NEXT: %load = load float, ptr %ptr, align 4, !nofpclass !7
define float @md_is_out_of_bounds(ptr %ptr) {
  %load = load float, ptr %ptr, align 4, !nofpclass !{i32 1024}
  ret float %load
}

declare float @func()

; CHECK: nofpclass is only for loads
; CHECK-NEXT: %result = call float @func(), !nofpclass !8
define float @not_load(ptr %ptr) {
  %result = call float @func(), !nofpclass !{i32 3}
  ret float %result
}

; CHECK: nofpclass only applies to floating-point typed loads
; CHECK-NEXT: %load = load i32, ptr %ptr, align 4, !nofpclass !8
define i32 @load_int(ptr %ptr) {
  %load = load i32, ptr %ptr, align 4, !nofpclass !{i32 3}
  ret i32 %load
}

; CHECK: nofpclass only applies to floating-point typed loads
; CHECK-NEXT: %load = load <2 x i32>, ptr %ptr, align 8, !nofpclass !8
define <2 x i32> @load_int_vec(ptr %ptr) {
  %load = load <2 x i32>, ptr %ptr, align 8, !nofpclass !{i32 3}
  ret <2 x i32> %load
}

%struct = type { i32, float }

; CHECK: nofpclass only applies to floating-point typed loads
; CHECK-NEXT: %load = load %struct, ptr %ptr, align 4, !nofpclass !8
define %struct @load_hetero_struct(ptr %ptr) {
  %load = load %struct, ptr %ptr, align 4, !nofpclass !{i32 3}
  ret %struct %load
}
