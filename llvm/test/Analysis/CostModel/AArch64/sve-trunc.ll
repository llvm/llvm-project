; RUN: opt -mtriple=aarch64-linux-gnu -mattr=+sve -cost-model -analyze < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

define void @sve_truncs() {
  ;CHECK-LABEL: 'sve_truncs'
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %trunc_v2i16_to_i1 = trunc <vscale x 2 x i16> undef to <vscale x 2 x i1>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %trunc_v2i32_to_i1 = trunc <vscale x 2 x i32> undef to <vscale x 2 x i1>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %trunc_v2i64_to_i1 = trunc <vscale x 2 x i64> undef to <vscale x 2 x i1>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %trunc_v4i16_to_i1 = trunc <vscale x 4 x i16> undef to <vscale x 4 x i1>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %trunc_v4i32_to_i1 = trunc <vscale x 4 x i32> undef to <vscale x 4 x i1>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %trunc_v4i64_to_i1 = trunc <vscale x 4 x i64> undef to <vscale x 4 x i1>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %trunc_v8i16_to_i1 = trunc <vscale x 8 x i16> undef to <vscale x 8 x i1>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %trunc_v8i32_to_i1 = trunc <vscale x 8 x i32> undef to <vscale x 8 x i1>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 5 for instruction:   %trunc_v8i64_to_i1 = trunc <vscale x 8 x i64> undef to <vscale x 8 x i1>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %trunc_v2i32_to_i16 = trunc <vscale x 2 x i32> undef to <vscale x 2 x i16>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %trunc_v2i64_to_i32 = trunc <vscale x 2 x i64> undef to <vscale x 2 x i32>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %trunc_v4i32_to_i16 = trunc <vscale x 4 x i32> undef to <vscale x 4 x i16>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %trunc_v4i64_to_i32 = trunc <vscale x 4 x i64> undef to <vscale x 4 x i32>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 3 for instruction:   %trunc_v8i32_to_i16 = trunc <vscale x 8 x i32> undef to <vscale x 8 x i16>
  ;CHECK-NEXT: Cost Model: Found an estimated cost of 6 for instruction:   %trunc_v8i64_to_i32 = trunc <vscale x 8 x i64> undef to <vscale x 8 x i32>
  %trunc_v2i16_to_i1  = trunc <vscale x 2 x i16> undef to <vscale x 2 x i1>
  %trunc_v2i32_to_i1  = trunc <vscale x 2 x i32> undef to <vscale x 2 x i1>
  %trunc_v2i64_to_i1  = trunc <vscale x 2 x i64> undef to <vscale x 2 x i1>

  %trunc_v4i16_to_i1  = trunc <vscale x 4 x i16> undef to <vscale x 4 x i1>
  %trunc_v4i32_to_i1  = trunc <vscale x 4 x i32> undef to <vscale x 4 x i1>
  %trunc_v4i64_to_i1  = trunc <vscale x 4 x i64> undef to <vscale x 4 x i1>

  %trunc_v8i16_to_i1  = trunc <vscale x 8 x i16> undef to <vscale x 8 x i1>
  %trunc_v8i32_to_i1  = trunc <vscale x 8 x i32> undef to <vscale x 8 x i1>
  %trunc_v8i64_to_i1  = trunc <vscale x 8 x i64> undef to <vscale x 8 x i1>

  %trunc_v2i32_to_i16 = trunc <vscale x 2 x i32> undef to <vscale x 2 x i16>
  %trunc_v2i64_to_i32 = trunc <vscale x 2 x i64> undef to <vscale x 2 x i32>

  %trunc_v4i32_to_i16 = trunc <vscale x 4 x i32> undef to <vscale x 4 x i16>
  %trunc_v4i64_to_i32 = trunc <vscale x 4 x i64> undef to <vscale x 4 x i32>

  %trunc_v8i32_to_i16 = trunc <vscale x 8 x i32> undef to <vscale x  8 x i16>
  %trunc_v8i64_to_i32 = trunc <vscale x 8 x i64> undef to <vscale x  8 x i32>

  ret void
}
