; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; CHECK: .visible .global .align 4 .u32 g = 42;
; CHECK: .visible .global .align 1 .b8 ga[4] = {0, 1, 2, 3};
; CHECK: .visible .global .align 8 .u64 g2 = generic(g);
; CHECK: .visible .global .align 8 .u64 g3 = g;
; CHECK: .visible .global .align 8 .u64 g4[2] = {0, generic(g)};
; CHECK: .visible .global .align 8 .u64 g5[2] = {0, generic(g)+8};

@g = addrspace(1) global i32 42
@ga = addrspace(1) global [4 x i8] c"\00\01\02\03"
@g2 = addrspace(1) global ptr addrspacecast (ptr addrspace(1) @g to ptr)
@g3 = addrspace(1) global ptr addrspace(1) @g
@g4 = constant {ptr, ptr} {ptr null, ptr addrspacecast (ptr addrspace(1) @g to ptr)}
@g5 = constant {ptr, ptr} {ptr null, ptr addrspacecast (ptr addrspace(1) getelementptr (i32, ptr addrspace(1) @g, i32 2) to ptr)}

; CHECK: .visible .global .align 8 .u64 g6 = generic(ga)+2;
@g6 = addrspace(1) global ptr getelementptr inbounds (
  [4 x i8], ptr addrspacecast (ptr addrspace(1) @ga to ptr),
  i32 0, i32 2
)

; CHECK: .visible .global .align 8 .u64 g7 = generic(g);
@g7 = addrspace(1) global ptr addrspacecast (
  ptr addrspace(1) @g
  to ptr
)

; CHECK: .visible .global .align 8 .u64 g8[2] = {0, g};
@g8 = addrspace(1) global [2 x ptr addrspace(1)] [ptr addrspace(1) null, ptr addrspace(1) @g]

; CHECK: .visible .global .align 8 .u64 g9[2] = {0, generic(g)};
@g9 = addrspace(1) global [2 x ptr] [
  ptr null,
  ptr addrspacecast (ptr addrspace(1) @g to ptr)
]

; CHECK: .visible .global .align 8 .u64 g10[2] = {0, g};
@g10 = addrspace(1) global [2 x ptr addrspace(1)] [
  ptr addrspace(1) null,
  ptr addrspace(1) @g
]

; CHECK: .visible .global .align 8 .u64 g11[2] = {0, generic(g)};
@g11 = addrspace(1) global [2 x ptr] [
  ptr null,
  ptr addrspacecast (ptr addrspace(1) @g to ptr)
]
