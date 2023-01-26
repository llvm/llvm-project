; XFAIL: *
; RUN: llvm-diff %s %s

define void @func() {
entry:
  %gep0 = getelementptr inbounds i8, ptr addrspace(4) null, i64 12
  %gep1 = getelementptr i8, ptr addrspace(4) null, i64 4
  ret void
}

uselistorder ptr addrspace(4) null, { 1, 0 }
