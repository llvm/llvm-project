; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; TODO: This test currently fails with LLVM_ENABLE_EXPENSIVE_CHECKS enabled
; XFAIL: expensive_checks

; CHECK-SPIRV-DAG: %[[#]] = OpGroupAsyncCopy %[[#]] %[[#Scope:]]
; CHECK-SPIRV-DAG: %[[#Scope]] = OpConstant %[[#]]

%opencl.event_t = type opaque

define spir_kernel void @test_fn(<2 x i8> addrspace(1)* %src, <2 x i8> addrspace(1)* %dst, <2 x i8> addrspace(3)* %localBuffer, i32 %copiesPerWorkgroup, i32 %copiesPerWorkItem) {
entry:
  %src.addr = alloca <2 x i8> addrspace(1)*, align 4
  %dst.addr = alloca <2 x i8> addrspace(1)*, align 4
  %localBuffer.addr = alloca <2 x i8> addrspace(3)*, align 4
  %copiesPerWorkgroup.addr = alloca i32, align 4
  %copiesPerWorkItem.addr = alloca i32, align 4
  %i = alloca i32, align 4
  %event = alloca %opencl.event_t*, align 4
  store <2 x i8> addrspace(1)* %src, <2 x i8> addrspace(1)** %src.addr, align 4
  store <2 x i8> addrspace(1)* %dst, <2 x i8> addrspace(1)** %dst.addr, align 4
  store <2 x i8> addrspace(3)* %localBuffer, <2 x i8> addrspace(3)** %localBuffer.addr, align 4
  store i32 %copiesPerWorkgroup, i32* %copiesPerWorkgroup.addr, align 4
  store i32 %copiesPerWorkItem, i32* %copiesPerWorkItem.addr, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4
  %1 = load i32, i32* %copiesPerWorkItem.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %call = call spir_func i32 @_Z12get_local_idj(i32 0)
  %2 = load i32, i32* %copiesPerWorkItem.addr, align 4
  %mul = mul i32 %call, %2
  %3 = load i32, i32* %i, align 4
  %add = add i32 %mul, %3
  %4 = load <2 x i8> addrspace(3)*, <2 x i8> addrspace(3)** %localBuffer.addr, align 4
  %arrayidx = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(3)* %4, i32 %add
  store <2 x i8> zeroinitializer, <2 x i8> addrspace(3)* %arrayidx, align 2
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, i32* %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  call spir_func void @_Z7barrierj(i32 1)
  store i32 0, i32* %i, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc12, %for.end
  %6 = load i32, i32* %i, align 4
  %7 = load i32, i32* %copiesPerWorkItem.addr, align 4
  %cmp2 = icmp slt i32 %6, %7
  br i1 %cmp2, label %for.body3, label %for.end14

for.body3:                                        ; preds = %for.cond1
  %call4 = call spir_func i32 @_Z13get_global_idj(i32 0)
  %8 = load i32, i32* %copiesPerWorkItem.addr, align 4
  %mul5 = mul i32 %call4, %8
  %9 = load i32, i32* %i, align 4
  %add6 = add i32 %mul5, %9
  %10 = load <2 x i8> addrspace(1)*, <2 x i8> addrspace(1)** %src.addr, align 4
  %arrayidx7 = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(1)* %10, i32 %add6
  %11 = load <2 x i8>, <2 x i8> addrspace(1)* %arrayidx7, align 2
  %call8 = call spir_func i32 @_Z12get_local_idj(i32 0)
  %12 = load i32, i32* %copiesPerWorkItem.addr, align 4
  %mul9 = mul i32 %call8, %12
  %13 = load i32, i32* %i, align 4
  %add10 = add i32 %mul9, %13
  %14 = load <2 x i8> addrspace(3)*, <2 x i8> addrspace(3)** %localBuffer.addr, align 4
  %arrayidx11 = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(3)* %14, i32 %add10
  store <2 x i8> %11, <2 x i8> addrspace(3)* %arrayidx11, align 2
  br label %for.inc12

for.inc12:                                        ; preds = %for.body3
  %15 = load i32, i32* %i, align 4
  %inc13 = add nsw i32 %15, 1
  store i32 %inc13, i32* %i, align 4
  br label %for.cond1

for.end14:                                        ; preds = %for.cond1
  call spir_func void @_Z7barrierj(i32 1)
  %16 = load <2 x i8> addrspace(1)*, <2 x i8> addrspace(1)** %dst.addr, align 4
  %17 = load i32, i32* %copiesPerWorkgroup.addr, align 4
  %call15 = call spir_func i32 @_Z12get_group_idj(i32 0)
  %mul16 = mul i32 %17, %call15
  %add.ptr = getelementptr inbounds <2 x i8>, <2 x i8> addrspace(1)* %16, i32 %mul16
  %18 = load <2 x i8> addrspace(3)*, <2 x i8> addrspace(3)** %localBuffer.addr, align 4
  %19 = load i32, i32* %copiesPerWorkgroup.addr, align 4
  %call17 = call spir_func %opencl.event_t* @_Z21async_work_group_copyPU3AS1Dv2_cPKU3AS3S_j9ocl_event(<2 x i8> addrspace(1)* %add.ptr, <2 x i8> addrspace(3)* %18, i32 %19, %opencl.event_t* null)
  store %opencl.event_t* %call17, %opencl.event_t** %event, align 4
  %20 = addrspacecast %opencl.event_t** %event to %opencl.event_t* addrspace(4)*
  call spir_func void @_Z17wait_group_eventsiPU3AS49ocl_event(i32 1, %opencl.event_t* addrspace(4)* %20)
  ret void
}

declare spir_func i32 @_Z12get_local_idj(i32)

declare spir_func void @_Z7barrierj(i32)

declare spir_func i32 @_Z13get_global_idj(i32)

declare spir_func %opencl.event_t* @_Z21async_work_group_copyPU3AS1Dv2_cPKU3AS3S_j9ocl_event(<2 x i8> addrspace(1)*, <2 x i8> addrspace(3)*, i32, %opencl.event_t*)

declare spir_func i32 @_Z12get_group_idj(i32)

declare spir_func void @_Z17wait_group_eventsiPU3AS49ocl_event(i32, %opencl.event_t* addrspace(4)*)
