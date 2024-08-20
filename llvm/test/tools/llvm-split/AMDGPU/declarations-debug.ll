; RUN: llvm-split -o %t %s -j 2 -mtriple amdgcn-amd-amdhsa --debug

; REQUIRES: asserts

; CHECK: --Partitioning Starts--
; CHECK: P0 has a total cost of 0 (0.00% of source module)
; CHECK: P1 has a total cost of 0 (0.00% of source module)
; CHECK: --Partitioning Done--

declare void @A()

declare void @B()
