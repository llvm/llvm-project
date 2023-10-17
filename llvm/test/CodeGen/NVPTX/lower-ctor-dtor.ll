; RUN: opt -S -mtriple=nvptx64-- -nvptx-lower-ctor-dtor < %s | FileCheck %s
; RUN: opt -S -mtriple=nvptx64-- -passes=nvptx-lower-ctor-dtor < %s | FileCheck %s
; RUN: opt -S -mtriple=nvptx64-- -passes=nvptx-lower-ctor-dtor \
; RUN:     -nvptx-lower-global-ctor-dtor-id=unique_id < %s | FileCheck %s --check-prefix=GLOBAL

; Make sure we get the same result if we run multiple times
; RUN: opt -S -mtriple=nvptx64-- -passes=nvptx-lower-ctor-dtor,nvptx-lower-ctor-dtor < %s | FileCheck %s
; RUN: llc -nvptx-lower-global-ctor-dtor -mtriple=nvptx64-amd-amdhsa -mcpu=sm_70 -filetype=asm -o - < %s | FileCheck %s -check-prefix=VISIBILITY

@llvm.global_ctors = appending addrspace(1) global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @foo, ptr null }]
@llvm.global_dtors = appending addrspace(1) global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @bar, ptr null }]

; CHECK-NOT: @llvm.global_ctors
; CHECK-NOT: @llvm.global_dtors

; CHECK: @__init_array_object_foo_[[HASH:[0-9a-f]+]]_1 = protected addrspace(4) constant ptr @foo, section ".init_array.1"
; CHECK: @__fini_array_object_bar_[[HASH:[0-9a-f]+]]_1 = protected addrspace(4) constant ptr @bar, section ".fini_array.1"
; CHECK: @llvm.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(4) @__init_array_object_foo_[[HASH]]_1 to ptr), ptr addrspacecast (ptr addrspace(4) @__fini_array_object_bar_[[HASH]]_1 to ptr)], section "llvm.metadata"
; GLOBAL: @__init_array_object_foo_unique_id_1 = protected addrspace(4) constant ptr @foo, section ".init_array.1"
; GLOBAL: @__fini_array_object_bar_unique_id_1 = protected addrspace(4) constant ptr @bar, section ".fini_array.1"
; GLOBAL: @llvm.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(4) @__init_array_object_foo_unique_id_1 to ptr), ptr addrspacecast (ptr addrspace(4) @__fini_array_object_bar_unique_id_1 to ptr)], section "llvm.metadata"

; VISIBILITY: .visible .const .align 8 .u64 __init_array_object_foo_[[HASH:[0-9a-f]+]]_1 = foo;
; VISIBILITY: .visible .const .align 8 .u64 __fini_array_object_bar_[[HASH:[0-9a-f]+]]_1 = bar;

define internal void @foo() {
  ret void
}

define internal void @bar() {
  ret void
}
