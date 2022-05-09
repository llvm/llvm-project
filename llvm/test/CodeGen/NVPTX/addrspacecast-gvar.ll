; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}

; CHECK: .visible .global .align 4 .u32 g = 42;
; CHECK: .visible .global .align 4 .u32 g2 = generic(g);
; CHECK: .visible .global .align 4 .u32 g3 = g;
; CHECK: .visible .global .align 8 .u32 g4[2] = {0, generic(g)};
; CHECK: .visible .global .align 8 .u32 g5[2] = {0, generic(g)+8};

@g = addrspace(1) global i32 42
@g2 = addrspace(1) global i32* addrspacecast (i32 addrspace(1)* @g to i32*)
@g3 = addrspace(1) global i32 addrspace(1)* @g
@g4 = constant {i32*, i32*} {i32* null, i32* addrspacecast (i32 addrspace(1)* @g to i32*)}
@g5 = constant {i32*, i32*} {i32* null, i32* addrspacecast (i32 addrspace(1)* getelementptr (i32, i32 addrspace(1)* @g, i32 2) to i32*)}
