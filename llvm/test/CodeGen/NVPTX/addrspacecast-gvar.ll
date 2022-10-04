; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}

; CHECK: .visible .global .align 4 .u32 g = 42;
; CHECK: .visible .global .align 1 .b8 ga[4] = {0, 1, 2, 3};
; CHECK: .visible .global .align 4 .u32 g2 = generic(g);
; CHECK: .visible .global .align 4 .u32 g3 = g;
; CHECK: .visible .global .align 8 .u32 g4[2] = {0, generic(g)};
; CHECK: .visible .global .align 8 .u32 g5[2] = {0, generic(g)+8};

@g = addrspace(1) global i32 42
@ga = addrspace(1) global [4 x i8] c"\00\01\02\03"
@g2 = addrspace(1) global i32* addrspacecast (i32 addrspace(1)* @g to i32*)
@g3 = addrspace(1) global i32 addrspace(1)* @g
@g4 = constant {i32*, i32*} {i32* null, i32* addrspacecast (i32 addrspace(1)* @g to i32*)}
@g5 = constant {i32*, i32*} {i32* null, i32* addrspacecast (i32 addrspace(1)* getelementptr (i32, i32 addrspace(1)* @g, i32 2) to i32*)}

; CHECK: .visible .global .align 4 .u32 g6 = generic(ga)+2;
@g6 = addrspace(1) global i8* getelementptr inbounds (
  [4 x i8], [4 x i8]* addrspacecast ([4 x i8] addrspace(1)* @ga to [4 x i8]*),
  i32 0, i32 2
)

; CHECK: .visible .global .align 4 .u32 g7 = generic(g);
@g7 = addrspace(1) global i8* addrspacecast (
  i8 addrspace(1)* bitcast (i32 addrspace(1)* @g to i8 addrspace(1)*)
  to i8*
)

; CHECK: .visible .global .align 4 .u32 g8[2] = {0, g};
@g8 = addrspace(1) global [2 x i32 addrspace(1)*] [i32 addrspace(1)* null, i32 addrspace(1)* @g]

; CHECK: .visible .global .align 4 .u32 g9[2] = {0, generic(g)};
@g9 = addrspace(1) global [2 x i32*] [
  i32* null,
  i32* addrspacecast (i32 addrspace(1)* @g to i32*)
]

; CHECK: .visible .global .align 4 .u32 g10[2] = {0, g};
@g10 = addrspace(1) global [2 x i8 addrspace(1)*] [
  i8 addrspace(1)* null,
  i8 addrspace(1)* bitcast (i32 addrspace(1)* @g to i8 addrspace(1)*)
]

; CHECK: .visible .global .align 4 .u32 g11[2] = {0, generic(g)};
@g11 = addrspace(1) global [2 x i8*] [
  i8* null,
  i8* bitcast (i32* addrspacecast (i32 addrspace(1)* @g to i32*) to i8*)
]
