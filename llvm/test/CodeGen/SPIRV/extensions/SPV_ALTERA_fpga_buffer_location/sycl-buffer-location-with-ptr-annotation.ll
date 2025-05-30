; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_ALTERA_fpga_buffer_location %s -o %t.spt
; RUN: FileCheck %s --input-file=%t.spt
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_ALTERA_fpga_buffer_location %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpDecorate %[[#Ptr1:]] BufferLocationALTERA 0
; CHECK-DAG: OpDecorate %[[#Ptr2:]] BufferLocationALTERA 0

; CHECK-DAG: %[[#Ptr1]] = OpLoad %[[#]] 
; CHECK-DAG: OpReturnValue %[[#Ptr1]]

; CHECK-DAG: %[[#Bitcast:]] = OpBitcast %[[#]] %[[#]]
; CHECK-DAG: %[[#Ptr2]] = OpInBoundsPtrAccessChain %[[#]] %[[#Bitcast]] %[[#]] %[[#]]
; CHECK-DAG: OpReturnValue %[[#Ptr2]]

%struct.MyIP = type <{ ptr addrspace(4), i32, [4 x i8] }>

$_ZNK4MyIPclEv = comdat any
$_Z8annotateIiEPT_S1_ = comdat any
$_Z9annotate2IiEPT_S1_ = comdat any

@.str = private unnamed_addr addrspace(1) constant [16 x i8] c"sycl-properties\00", section "llvm.metadata"
@.str.1 = private unnamed_addr addrspace(1) constant [9 x i8] c"test.cpp\00", section "llvm.metadata"
@.str.2 = private unnamed_addr addrspace(1) constant [21 x i8] c"sycl-buffer-location\00", section "llvm.metadata"
@.str.3 = private unnamed_addr addrspace(1) constant [2 x i8] c"0\00", section "llvm.metadata"
@.args = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) @.str.2, ptr addrspace(1) @.str.3 }, section "llvm.metadata"
@.str.4 = private unnamed_addr addrspace(1) constant [11 x i8] c"{5921:\220\22}\00", section "llvm.metadata"

define linkonce_odr dso_local spir_func void @_ZNK4MyIPclEv(ptr addrspace(4) %this) comdat align 2 !srcloc !8 {
entry:
  %call1 = call spir_func noundef ptr addrspace(4) @_Z8annotateIiEPT_S1_(ptr addrspace(4) noundef %this)
  %call2 = call spir_func noundef ptr addrspace(4) @_Z9annotate2IiEPT_S1_(ptr addrspace(4) noundef %this)
  ret void
}

define linkonce_odr dso_local spir_func noundef ptr addrspace(4) @_Z8annotateIiEPT_S1_(ptr addrspace(4) noundef %ptr) comdat !srcloc !9 {
entry:
  %retval = alloca ptr addrspace(4), align 8
  %ptr.addr = alloca ptr addrspace(4), align 8
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %ptr.addr.ascast = addrspacecast ptr %ptr.addr to ptr addrspace(4)
  store ptr addrspace(4) %ptr, ptr addrspace(4) %ptr.addr.ascast, align 8
  %0 = load ptr addrspace(4), ptr addrspace(4) %ptr.addr.ascast, align 8
  %1 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %0, ptr addrspace(1) @.str.4, ptr addrspace(1) @.str.1, i32 25, ptr addrspace(1) null)
  ret ptr addrspace(4) %1
}

define linkonce_odr dso_local spir_func noundef ptr addrspace(4) @_Z9annotate2IiEPT_S1_(ptr addrspace(4) noundef %ptr) comdat !srcloc !9 {
entry:
  %retval = alloca ptr addrspace(4), align 8
  %ptr.addr = alloca ptr addrspace(4), align 8
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %ptr.addr.ascast = addrspacecast ptr %ptr.addr to ptr addrspace(4)
  store ptr addrspace(4) %ptr, ptr addrspace(4) %ptr.addr.ascast, align 8
  %0 = load ptr addrspace(4), ptr addrspace(4) %ptr.addr.ascast, align 8
  %1 = getelementptr inbounds %struct.MyIP, ptr addrspace(4) %0, i32 0, i32 0
  %2 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %1, ptr addrspace(1) @.str.4, ptr addrspace(1) @.str.1, i32 25, ptr addrspace(1) null)
  ret ptr addrspace(4) %2
}

declare ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1))

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"Intel(R) oneAPI DPC++/C++ Compiler 2024.2.0 (2024.x.0.YYYYMMDD)"}
!5 = !{i32 717}
!6 = !{i32 -1, i32 -1}
!7 = !{}
!8 = !{i32 1004}
!9 = !{i32 563}
