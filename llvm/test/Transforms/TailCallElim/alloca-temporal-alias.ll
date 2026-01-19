; RUN: opt -passes=tailcallelim -S < %s | FileCheck %s

; Test that Tail Call Elimination can correctly identify when an alloca's 
; address is stored in another alloca but does not escape.
; The analysis should recognize that if all loads from the storage alloca 
; occur before the recursive call, the original alloca remains safe for TRE.

define i32 @test_temporary_alias(i32 %n, i32 %acc) {
; CHECK-LABEL: @test_temporary_alias
; CHECK: tailrecurse:
entry:
  %local_var = alloca i32
  %ptr_storage = alloca ptr
  
  ; Store the address of the local variable into a temporary storage alloca.
  store ptr %local_var, ptr %ptr_storage
  
  ; Access the stored address locally.
  %addr = load ptr, ptr %ptr_storage
  store i32 42, ptr %addr
  
  ; At this point, %ptr_storage is no longer accessed. 
  ; The flow-sensitive analysis using DominatorTree should prove that 
  ; %local_var does not escape through %ptr_storage in any path reaching 
  ; the recursive call.

  %cond = icmp eq i32 %n, 0
  br i1 %cond, label %exit, label %recurse

recurse:
  %n_dec = sub i32 %n, 1
  %acc_inc = add i32 %acc, 1
  
  ; TRE should be applied here because the address of %local_var 
  ; is not captured by any instruction dominating this call.
  %res = call i32 @test_temporary_alias(i32 %n_dec, i32 %acc_inc)
  ret i32 %res

exit:
  ret i32 %acc
}
