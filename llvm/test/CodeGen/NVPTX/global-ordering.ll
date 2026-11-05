; RUN: llc < %s -mtriple=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=PTX64
; RUN: %if ptxas-ptr32 %{ llc < %s -mtriple=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; Make sure we emit these globals in def-use order


; PTX32:      .visible .global .align 1 .u8 a = 2;
; PTX32-NEXT: .visible .global .align 4 .u32 a2 = a;
; PTX64:      .visible .global .align 1 .u8 a = 2;
; PTX64-NEXT: .visible .global .align 8 .u64 a2 = a;
@a2 = addrspace(1) global ptr addrspace(1) @a
@a = addrspace(1) global i8 2


; PTX32:      .visible .global .align 1 .u8 b = 1;
; PTX32-NEXT: .visible .global .align 4 .u32 b2[2] = {b, b};
; PTX64:      .visible .global .align 1 .u8 b = 1;
; PTX64-NEXT: .visible .global .align 8 .u64 b2[2] = {b, b};
@b2 = addrspace(1) global [2 x ptr addrspace(1)] [ptr addrspace(1) @b, ptr addrspace(1) @b]
@b = addrspace(1) global i8 1


; A GEP initializer is emitted as its base plus a constant byte offset. A
; symbol nested in an index expression is not emitted and must not create a
; false self-cycle. The zero-sized element makes the computed offset zero.
; Consumer appears first in IR to check dependency-first emission.
;
; PTX32:      .global .align 4 .u32 gep_index_base = 7;
; PTX32-NEXT: .global .align 4 .u32 gep_index = gep_index_base;
; PTX64:      .global .align 4 .u32 gep_index_base = 7;
; PTX64-NEXT: .global .align 8 .u64 gep_index = gep_index_base;
%empty = type {}
@gep_index = internal addrspace(1) global ptr addrspace(1) getelementptr (
    %empty, ptr addrspace(1) @gep_index_base,
    i64 ptrtoint (ptr addrspace(1) @gep_index to i64))
@gep_index_base = internal addrspace(1) global i32 7


; A Function is also referenced as a single symbol. Its personality is not
; part of the global initializer and must not create a false self-cycle.
; PTX32: .global .align 4 .u32 function_leaf_ref = function_leaf;
; PTX64: .global .align 8 .u64 function_leaf_ref = function_leaf;
@function_leaf_ref = internal addrspace(1) global ptr @function_leaf


; Emit a global aggregate with a field computed from the address of another
; global.
@g = addrspace(1) global i8 0

; PTX64: .visible .global .align 8 .u64 sadd[2] = {g+4, 7};
@sadd = addrspace(1) global { i64, i64 } { i64 add (i64 ptrtoint (ptr addrspace(1) @g to i64), i64 4), i64 7 }

; Self-references are emitted as a declaration followed by the definition.
; PTX32:      .extern .global .align 4 .u32 self;
; PTX32-NEXT: .visible .global .align 4 .u32 self = self;
; PTX64:      .extern .global .align 8 .u64 self;
; PTX64-NEXT: .visible .global .align 8 .u64 self = self;
@self = addrspace(1) global ptr addrspace(1) @self

define internal void @function_leaf()
    personality ptr addrspace(1) @function_leaf_ref {
  ret void
}
