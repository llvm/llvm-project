; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-lower-ctor-dtor %s | FileCheck %s

; Make sure we emit code for constructor entries that aren't direct
; function calls.

; Check a constructor that's an alias, and an integer literal.
@llvm.global_ctors = appending addrspace(1) global [2 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 1, ptr @foo.alias, i8* null },
  { i32, ptr, ptr } { i32 1, ptr inttoptr (i64 4096 to ptr), i8* null }
]

; Check a constantexpr addrspacecast
@llvm.global_dtors = appending addrspace(1) global [1 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 1, ptr addrspacecast (ptr addrspace(1) @bar to ptr), i8* null }
]

@foo.alias = hidden alias void (), ptr @foo

;.
; CHECK: @__init_array_start = external addrspace(1) constant [0 x ptr addrspace(1)]
; CHECK: @__init_array_end = external addrspace(1) constant [0 x ptr addrspace(1)]
; CHECK: @__fini_array_start = external addrspace(1) constant [0 x ptr addrspace(1)]
; CHECK: @__fini_array_end = external addrspace(1) constant [0 x ptr addrspace(1)]
; CHECK: @llvm.used = appending global [2 x ptr] [ptr @amdgcn.device.init, ptr @amdgcn.device.fini], section "llvm.metadata"
; CHECK: @foo.alias = hidden alias void (), ptr @foo
;.
define void @foo() {
; CHECK-LABEL: @foo(
; CHECK-NEXT:    ret void
;
  ret void
}

define void @bar() addrspace(1) {
; CHECK-LABEL: @bar(
; CHECK-NEXT:    ret void
;
  ret void
}

; CHECK-LABEL: define weak_odr amdgpu_kernel void @amdgcn.device.init()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 icmp ne (ptr addrspace(1) @__init_array_start, ptr addrspace(1) @__init_array_end), label [[WHILE_ENTRY:%.*]], label [[WHILE_END:%.*]]
; CHECK:       while.entry:
; CHECK-NEXT:    [[PTR:%.*]] = phi ptr addrspace(1) [ @__init_array_start, [[ENTRY:%.*]] ], [ [[NEXT:%.*]], [[WHILE_ENTRY]] ]
; CHECK-NEXT:    [[CALLBACK:%.*]] = load ptr, ptr addrspace(1) [[PTR]], align 8
; CHECK-NEXT:    call void [[CALLBACK]]()
; CHECK-NEXT:    [[NEXT]] = getelementptr ptr addrspace(1), ptr addrspace(1) [[PTR]], i64 1
; CHECK-NEXT:    [[END:%.*]] = icmp eq ptr addrspace(1) [[NEXT]], @__init_array_end
; CHECK-NEXT:    br i1 [[END]], label [[WHILE_END]], label [[WHILE_ENTRY]]
; CHECK:       while.end:
; CHECK-NEXT:    ret void

; CHECK-LABEL: define weak_odr amdgpu_kernel void @amdgcn.device.fini()
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 icmp ne (ptr addrspace(1) @__fini_array_start, ptr addrspace(1) @__fini_array_end), label [[WHILE_ENTRY:%.*]], label [[WHILE_END:%.*]]
; CHECK:       while.entry:
; CHECK-NEXT:    [[PTR:%.*]] = phi ptr addrspace(1) [ @__fini_array_start, [[ENTRY:%.*]] ], [ [[NEXT:%.*]], [[WHILE_ENTRY]] ]
; CHECK-NEXT:    [[CALLBACK:%.*]] = load ptr, ptr addrspace(1) [[PTR]], align 8
; CHECK-NEXT:    call void [[CALLBACK]]()
; CHECK-NEXT:    [[NEXT]] = getelementptr ptr addrspace(1), ptr addrspace(1) [[PTR]], i64 1
; CHECK-NEXT:    [[END:%.*]] = icmp eq ptr addrspace(1) [[NEXT]], @__fini_array_end
; CHECK-NEXT:    br i1 [[END]], label [[WHILE_END]], label [[WHILE_ENTRY]]
; CHECK:       while.end:
; CHECK-NEXT:    ret void

; CHECK: attributes #[[ATTR0:[0-9]+]] = { "amdgpu-flat-work-group-size"="1,1" "device-init" }
; CHECK: attributes #[[ATTR1:[0-9]+]] = { "amdgpu-flat-work-group-size"="1,1" "device-fini" }
