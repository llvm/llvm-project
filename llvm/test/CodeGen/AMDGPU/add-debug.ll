; RUN: llc < %s -march=amdgcn -mcpu=tahiti -debug
; RUN: llc < %s -march=amdgcn -mcpu=tonga -debug
; REQUIRES: asserts

; Check that SelectionDAGDumper does not crash on int_SI_if.
define amdgpu_kernel void @add64_in_branch(ptr addrspace(1) %out, ptr addrspace(1) %in, i64 %a, i64 %b, i64 %c) {
entry:
  %0 = icmp eq i64 %a, 0
  br i1 %0, label %if, label %else

if:
  %1 = load i64, ptr addrspace(1) %in
  br label %endif

else:
  %2 = add i64 %a, %b
  br label %endif

endif:
  %3 = phi i64 [%1, %if], [%2, %else]
  store i64 %3, ptr addrspace(1) %out
  ret void
}

