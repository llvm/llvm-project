; RUN: opt -passes=fix-irreducible,unify-loop-exits,structurizecfg -S %s | FileCheck %s
;
; Reduced from rocPRIM block_sort_kernel compiled with code coverage.
; The sort comparator `operator<` for custom_type uses nested short-circuit:
;   less<T>(x) || (equal_to<T>(x) && less<U>(y))
; With PGO instrumentation, each functor call gets a counter block,
; creating a deeply nested CFG that StructurizeCFG must linearize.
;
; The bug: hoistZeroCostElseBlockPhiValues() hoists shufflevector out of
; the "swap" block into a dominator, and simplifyHoistedPhis() then
; fills poison entries in Flow phis indiscriminately — causing the
; hoisted shufflevector result to reach the "no-swap" path, swapping
; two sort keys that should not be swapped.
;
; After structurizecfg, the shufflevector that performs the conditional
; swap must NOT appear before the first comparison block.

@counter_less = external addrspace(1) global [4 x i64]
@counter_eq = external addrspace(1) global [2 x i64]
@counter_swap = external addrspace(1) global [2 x i64]

; Two consecutive compare-and-swap iterations from a sorting network.
; Each iteration compares key_a vs key_b using:
;   less(a.x, b.x) || (equal(a.x, b.x) && less(a.y, b.y))
; If true, swap both keys and their index vector.
;
; iter1 compares keys[0] vs keys[1]
; iter2 compares keys[1] vs keys[2], using the (possibly swapped) result of iter1
;
; The values_vec (<4 x i32>) tracks the value indices associated with the keys.
; A shufflevector <0,2,1,3> swaps the middle two elements (the values for the two keys being compared).
define amdgpu_kernel void @sort_two_iters(
    i1 %do_iter1, i1 %do_iter2,
    <2 x float> %key_a, <2 x float> %key_b, <2 x float> %key_c,
    <4 x i32> %values_vec,
    ptr addrspace(1) %out_values, ptr addrspace(1) %out_key
) {
entry:
  %cnt_less0 = load i64, ptr addrspace(1) @counter_less, align 8
  %cnt_eq0 = load i64, ptr addrspace(1) @counter_eq, align 8
  %cnt_swap0 = load i64, ptr addrspace(1) @counter_swap, align 8
  br i1 %do_iter1, label %iter1.cmp_x, label %iter1.done

; --- Iteration 1: compare key_a vs key_b ---
iter1.cmp_x:
  ; PGO: counter for less<float>(a.x, b.x)
  %cnt_less1 = add i64 %cnt_less0, 1
  store i64 %cnt_less1, ptr addrspace(1) @counter_less, align 8
  %ax = extractelement <2 x float> %key_a, i64 0
  %bx = extractelement <2 x float> %key_b, i64 0
  %x_less = fcmp olt float %ax, %bx
  br i1 %x_less, label %iter1.do_swap, label %iter1.check_eq

iter1.check_eq:
  ; PGO: counter for equal_to<float>(a.x, b.x)
  %cnt_eq1 = add i64 %cnt_eq0, 1
  store i64 %cnt_eq1, ptr addrspace(1) @counter_eq, align 8
  %x_eq = fcmp oeq float %ax, %bx
  br i1 %x_eq, label %iter1.cmp_y, label %iter1.done

iter1.cmp_y:
  ; PGO: counter for less<float>(a.y, b.y)
  %cnt_less2 = add i64 %cnt_less0, 2
  store i64 %cnt_less2, ptr addrspace(1) @counter_less, align 8
  %ay = extractelement <2 x float> %key_a, i64 1
  %by = extractelement <2 x float> %key_b, i64 1
  %y_less = fcmp olt float %ay, %by
  br i1 %y_less, label %iter1.do_swap, label %iter1.done

iter1.do_swap:
  ; PGO: counter for swap
  %cnt_swap1 = add i64 %cnt_swap0, 1
  store i64 %cnt_swap1, ptr addrspace(1) @counter_swap, align 8
  ; Swap middle two elements of values_vec (the indices for the two compared keys)
  %swapped1 = shufflevector <4 x i32> %values_vec, <4 x i32> poison, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  br label %iter1.done

iter1.done:
  ; Select swapped or original values, and swapped keys
  %vals1 = phi <4 x i32> [ %swapped1, %iter1.do_swap ], [ %values_vec, %iter1.check_eq ], [ %values_vec, %iter1.cmp_y ], [ %values_vec, %entry ]
  %cnt_eq_out1 = phi i64 [ %cnt_eq0, %iter1.do_swap ], [ %cnt_eq1, %iter1.check_eq ], [ %cnt_eq1, %iter1.cmp_y ], [ %cnt_eq0, %entry ]
  %cnt_less_out1 = phi i64 [ %cnt_less1, %iter1.do_swap ], [ %cnt_less1, %iter1.check_eq ], [ %cnt_less2, %iter1.cmp_y ], [ %cnt_less0, %entry ]
  ; Swapped keys: if swap happened, key_a and key_b switch
  %key_lo1 = phi <2 x float> [ %key_b, %iter1.do_swap ], [ %key_a, %iter1.check_eq ], [ %key_a, %iter1.cmp_y ], [ %key_a, %entry ]
  %key_hi1 = phi <2 x float> [ %key_a, %iter1.do_swap ], [ %key_b, %iter1.check_eq ], [ %key_b, %iter1.cmp_y ], [ %key_b, %entry ]
  br i1 %do_iter2, label %iter2.cmp_x, label %iter2.done

; --- Iteration 2: compare key_hi1 (from iter1) vs key_c ---
iter2.cmp_x:
  ; PGO: counter for less<float>
  %cnt_less3 = add i64 %cnt_less_out1, 1
  store i64 %cnt_less3, ptr addrspace(1) @counter_less, align 8
  %cx = extractelement <2 x float> %key_c, i64 0
  %hi1x = extractelement <2 x float> %key_hi1, i64 0
  %x_less2 = fcmp olt float %hi1x, %cx
  br i1 %x_less2, label %iter2.do_swap, label %iter2.check_eq

iter2.check_eq:
  ; PGO: counter for equal_to<float>
  %cnt_eq2 = add i64 %cnt_eq_out1, 1
  store i64 %cnt_eq2, ptr addrspace(1) @counter_eq, align 8
  %x_eq2 = fcmp oeq float %hi1x, %cx
  br i1 %x_eq2, label %iter2.cmp_y, label %iter2.done

iter2.cmp_y:
  ; PGO: counter for less<float>
  %cnt_less4 = add i64 %cnt_less_out1, 2
  store i64 %cnt_less4, ptr addrspace(1) @counter_less, align 8
  %hi1y = extractelement <2 x float> %key_hi1, i64 1
  %cy = extractelement <2 x float> %key_c, i64 1
  %y_less2 = fcmp olt float %hi1y, %cy
  br i1 %y_less2, label %iter2.do_swap, label %iter2.done

iter2.do_swap:
  ; PGO: counter for swap
  %cnt_swap2 = add i64 %cnt_swap0, 2
  store i64 %cnt_swap2, ptr addrspace(1) @counter_swap, align 8
  ; Swap middle two elements of vals1 (the values that resulted from iter1)
  %swapped2 = shufflevector <4 x i32> %vals1, <4 x i32> poison, <4 x i32> <i32 1, i32 0, i32 2, i32 3>
  br label %iter2.done

iter2.done:
  %vals2 = phi <4 x i32> [ %swapped2, %iter2.do_swap ], [ %vals1, %iter2.check_eq ], [ %vals1, %iter2.cmp_y ], [ %vals1, %iter1.done ]
  store <4 x i32> %vals2, ptr addrspace(1) %out_values, align 16
  ret void
}

; After structurizecfg, verify that the shufflevector for iter1's swap
; is NOT hoisted into iter1.cmp_x. It must stay in iter1.do_swap (or
; a Flow block gated by the swap predicate) so that the no-swap path
; never sees the swapped values.
;
; CHECK-LABEL: @sort_two_iters
;
; The shufflevector must NOT appear in iter1.cmp_x (it would be hoisted there by the bug):
; CHECK:      iter1.cmp_x:
; CHECK-NOT:  shufflevector
; CHECK:      br i1
;
; It should remain in iter1.do_swap:
; CHECK:      iter1.do_swap:
; CHECK:      %swapped1 = shufflevector <4 x i32> %values_vec, <4 x i32> poison, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
;
; The vals1 phi at iter1.done must select between the Flow block result
; (which should carry both swapped and unswapped correctly) and the
; entry bypass. It must NOT directly carry the hoisted shufflevector result
; from a block that's reachable without going through the swap predicate.
; CHECK:      iter1.done:
; CHECK:      %vals1 = phi <4 x i32>
