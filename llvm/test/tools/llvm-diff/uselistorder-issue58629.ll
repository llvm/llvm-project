; RUN: llvm-diff %s %s | count 0
; Make sure there is no error produced by using uselistorder with two
; modules using the same constant in the same context.

define void @func() {
entry:
  %gep0 = getelementptr inbounds i8, ptr addrspace(4) null, i64 12
  %gep1 = getelementptr i8, ptr addrspace(4) null, i64 4
  ret void
}

uselistorder ptr addrspace(4) null, { 1, 0 }
