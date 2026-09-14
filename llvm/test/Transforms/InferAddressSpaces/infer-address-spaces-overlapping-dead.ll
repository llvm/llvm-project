; REQUIRES: x86-registered-target
; RUN: opt -disable-output -passes=infer-address-spaces \
; RUN:   -assume-default-is-flat-addrspace -mtriple=x86_64-unknown-unknown %s

; InferAddressSpaces can queue the same dead instruction more than once.
; Deleting an earlier entry invalidates later raw pointers, so the deletion
; worklist must track erased values safely.

@global = addrspace(1) global i32 0

define void @overlapping_dead_inferred_addresses() {
entry:
  br label %body

exit:
  %base = inttoptr i64 ptrtoint (ptr addrspacecast (ptr addrspace(1) @global to ptr) to i64) to ptr
  %unused = getelementptr i8, ptr %base, i64 0
  ret void

body:
  store i32 0, ptr inttoptr (i64 ptrtoint (ptr addrspace(1) @global to i64) to ptr), align 4
  br label %exit
}

uselistorder ptr addrspace(1) @global, { 1, 0 }
