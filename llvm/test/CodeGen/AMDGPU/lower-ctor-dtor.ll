; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-ctor-dtor < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-ctor-dtor < %s | FileCheck %s

; Make sure we get the same result if we run multiple times
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-ctor-dtor,amdgpu-lower-ctor-dtor < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readelf -s - 2>&1 | FileCheck %s -check-prefix=VISIBILITY
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readelf -S - 2>&1 | FileCheck %s -check-prefix=SECTION
; RUN: llc -mtriple=amdgcn-amd-amdhsa -amdgpu-lower-global-ctor-dtor=0 -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readelf -s - 2>&1 | FileCheck %s -check-prefix=DISABLED
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readelf --notes - 2>&1 | FileCheck %s -check-prefix=METADATA

@llvm.global_ctors = appending addrspace(1) global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @foo, ptr null }]
@llvm.global_dtors = appending addrspace(1) global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @bar, ptr null }]

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

; CHECK-NOT: amdgcn.device.

; VISIBILITY: FUNC   WEAK PROTECTED {{.*}} amdgcn.device.init
; VISIBILITY: OBJECT WEAK DEFAULT {{.*}} amdgcn.device.init.kd
; VISIBILITY: FUNC   WEAK PROTECTED {{.*}} amdgcn.device.fini
; VISIBILITY: OBJECT   WEAK DEFAULT {{.*}} amdgcn.device.fini.kd
; SECTION: .init_array.1     INIT_ARRAY      {{.*}} {{.*}} 000008 00  WA  0   0  8
; SECTION: .fini_array.1     FINI_ARRAY      {{.*}} {{.*}} 000008 00  WA  0   0  8
; DISABLED-NOT: FUNC   GLOBAL PROTECTED {{.*}} amdgcn.device.init
; DISABLED-NOT: OBJECT GLOBAL DEFAULT {{.*}} amdgcn.device.init.kd
; DISABLED-NOT: FUNC   GLOBAL PROTECTED {{.*}} amdgcn.device.fini
; DISABLED-NOT: OBJECT   GLOBAL DEFAULT {{.*}} amdgcn.device.fini.kd
; METADATA:  amdhsa.kernels:
; METADATA:    .kind:           init
; METADATA:    .max_flat_workgroup_size: 1
; METADATA:    .name:           amdgcn.device.init
; METADATA:    .symbol:         amdgcn.device.init.kd
; METADATA:    .kind:           fini
; METADATA:    .max_flat_workgroup_size: 1
; METADATA:    .name:           amdgcn.device.fini
; METADATA:    .symbol:         amdgcn.device.fini.kd

define internal void @foo() {
  ret void
}

define internal void @bar() {
  ret void
}

; CHECK: attributes #0 = { "amdgpu-flat-work-group-size"="1,1" "device-init" }
; CHECK: attributes #1 = { "amdgpu-flat-work-group-size"="1,1" "device-fini" }
