; RUN: opt -passes=tailcallelim -S < %s | FileCheck %s

; Test that Tail Call Elimination (TRE) can occur even when an alloca's address
; is stored in memory, provided that the DominatorTree-based analysis can prove 
; that the storage is local and the address does not escape into the recursive call.

define i32 @test_alloca_reuse(i32 %n, i32 %acc) {
; CHECK-LABEL: define i32 @test_alloca_reuse(i32 %n, i32 %acc)
; CHECK: entry:
; CHECK:   %temp_slot = alloca ptr
; CHECK:   br label %tailrecurse

; CHECK: tailrecurse:
; CHECK-NEXT: %n.tr = phi i32 [ %n, %entry ], [ %n_dec, %recurse ]
; CHECK-NEXT: %acc.tr = phi i32 [ %acc, %entry ], [ %acc_inc, %recurse ]

entry:
  %temp_slot = alloca ptr
  %dummy = alloca i32
  
  ; Store and immediately reload the address of 'dummy'.
  ; This analysis uses flow-sensitivity to prove that these memory operations
  ; occur before the recursive branch and do not result in a capture of 
  ; the alloca's address.
  store ptr %dummy, ptr %temp_slot
  %val = load ptr, ptr %temp_slot

  ; Check if the recursion depth has reached zero.
  %cond = icmp eq i32 %n, 0
  br i1 %cond, label %exit, label %recurse

recurse:
  ; Prepare arguments for the next iteration.
  %n_dec = sub i32 %n, 1
  %acc_inc = add i32 %acc, 1
  
  ; This call should be transformed into a branch back to 'tailrecurse'.
  ; PHI nodes will handle the state transition for %n and %acc.
  %res = call i32 @test_alloca_reuse(i32 %n_dec, i32 %acc_inc)
  ret i32 %res

exit:
  ; Return the accumulated result.
  ret i32 %acc
}
