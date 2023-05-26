; RUN: opt -S -mtriple=amdgcn--  -amdgpu-lower-ctor-dtor < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readelf -s - 2>&1 | FileCheck %s -check-prefix=CHECK-VIS

@llvm.global_ctors = appending addrspace(1) global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @foo, ptr null }, { i32, ptr, ptr } { i32 1, ptr @foo.5, ptr null }]
@llvm.global_dtors = appending addrspace(1) global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @bar, ptr null }, { i32, ptr, ptr } { i32 1, ptr @bar.5, ptr null }]

; CHECK: @__init_array_start = external addrspace(1) constant [0 x ptr addrspace(1)]
; CHECK: @__init_array_end = external addrspace(1) constant [0 x ptr addrspace(1)]
; CHECK: @__fini_array_start = external addrspace(1) constant [0 x ptr addrspace(1)]
; CHECK: @__fini_array_end = external addrspace(1) constant [0 x ptr addrspace(1)]
; CHECK: @llvm.used = appending global [2 x ptr] [ptr @amdgcn.device.init, ptr @amdgcn.device.fini]

; CHECK-LABEL: define weak_odr amdgpu_kernel void @amdgcn.device.init() #0
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

; CHECK-LABEL: define weak_odr amdgpu_kernel void @amdgcn.device.fini() #1
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

; CHECK-VIS: FUNC   WEAK PROTECTED {{.*}} amdgcn.device.init
; CHECK-VIS: OBJECT WEAK DEFAULT {{.*}} amdgcn.device.init.kd
; CHECK-VIS: FUNC   WEAK PROTECTED {{.*}} amdgcn.device.fini
; CHECK-VIS: OBJECT   WEAK DEFAULT {{.*}} amdgcn.device.fini.kd

define internal void @foo() {
  ret void
}

define internal void @bar() {
  ret void
}

define internal void @foo.5() {
  ret void
}

define internal void @bar.5() {
  ret void
}

; CHECK: attributes #0 = { "amdgpu-flat-work-group-size"="1,1" "device-init" }
; CHECK: attributes #1 = { "amdgpu-flat-work-group-size"="1,1" "device-fini" }
