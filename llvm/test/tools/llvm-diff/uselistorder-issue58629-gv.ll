; RUN: llvm-diff %s %s | count 0
; Make sure there is no error produced by using uselistorder with two
; modules using the same constant/global in the same context.

@gv = addrspace(4) global [2 x i64] zeroinitializer, align 16

define void @func() {
entry:
  %gep0 = getelementptr inbounds i8, ptr addrspace(4) @gv, i64 12
  %gep1 = getelementptr i8, ptr addrspace(4) @gv, i64 4
  ret void
}

uselistorder ptr addrspace(4) @gv, { 1, 0 }
