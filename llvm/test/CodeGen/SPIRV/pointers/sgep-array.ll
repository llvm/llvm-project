; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-pc-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-pc-vulkan1.3-library %s -o - -filetype=obj | spirv-val %}

%struct.S = type { i32, i32 }
%struct.S2 = type { i32, %struct.S, i32 }

; CHECK-DAG:           %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:          %[[#ulong:]] = OpTypeInt 64 0
; CHECK-DAG:        %[[#ulong_0:]] = OpConstant %[[#ulong]] 0
; CHECK-DAG:        %[[#ulong_1:]] = OpConstant %[[#ulong]] 1
; CHECK-DAG:        %[[#ulong_2:]] = OpConstant %[[#ulong]] 2
; CHECK-DAG:        %[[#ulong_3:]] = OpConstant %[[#ulong]] 3
; CHECK-DAG:         %[[#uint_0:]] = OpConstant %[[#uint]] 0
; CHECK-DAG:         %[[#uint_1:]] = OpConstant %[[#uint]] 1
; CHECK-DAG:         %[[#uint_5:]] = OpConstant %[[#uint]] 5
; CHECK-DAG:       %[[#ptr_uint:]] = OpTypePointer Function %[[#uint]]
; CHECK-DAG:     %[[#arr_uint_5:]] = OpTypeArray %[[#uint]] %[[#uint_5]]
; CHECK-DAG: %[[#ptr_arr_uint_5:]] = OpTypePointer Function %[[#arr_uint_5]]
; CHECK-DAG:              %[[#S:]] = OpTypeStruct %[[#uint]] %[[#uint]]
; CHECK-DAG:          %[[#ptr_S:]] = OpTypePointer Function %[[#S]]
; CHECK-DAG:        %[[#arr_S_5:]] = OpTypeArray %[[#S]] %[[#uint_5]]
; CHECK-DAG:    %[[#ptr_arr_S_5:]] = OpTypePointer Function %[[#arr_S_5]]
; CHECK-DAG:             %[[#S2:]] = OpTypeStruct %[[#uint]] %[[#S]] %[[#uint]]
; CHECK-DAG:         %[[#ptr_S2:]] = OpTypePointer Function %[[#S2]]
; CHECK-DAG:       %[[#arr_S2_5:]] = OpTypeArray %[[#S2]] %[[#uint_5]]
; CHECK-DAG:   %[[#ptr_arr_S2_5:]] = OpTypePointer Function %[[#arr_S2_5]]

define spir_func void @array_load_store(ptr %a) convergent {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %tmp = alloca [5 x i32], align 4
; CHECK:	%[[#tmp:]] = OpVariable %[[#ptr_arr_uint_5]] Function

  %1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([3 x i32]) %a, i64 2)
  %2 = load i32, ptr %1, align 4
; CHECK:	  %[[#A:]] = OpInBoundsAccessChain %[[#ptr_uint]] %[[#]] %[[#ulong_2]]
; CHECK:	  %[[#B:]] = OpLoad %[[#uint]] %[[#A]] Aligned 4

  %3 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([5 x i32]) %tmp, i64 3)
  store i32 %2, ptr %3, align 4
; CHECK:	  %[[#C:]] = OpInBoundsAccessChain %[[#ptr_uint]] %[[#tmp]] %[[#ulong_3]]
; CHECK:	             OpStore %[[#C]] %[[#B]] Aligned 4

  ret void
}

define spir_func void @array_struct_load_store(ptr %a) convergent {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %tmp = alloca [5 x %struct.S], align 4
; CHECK:	%[[#tmp:]] = OpVariable %[[#ptr_arr_S_5]] Function

  %1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([3 x %struct.S]) %a, i64 2)
  %2 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.S) %1, i64 1)
  %3 = load i32, ptr %2, align 4
; CHECK:	  %[[#A:]] = OpInBoundsAccessChain %[[#ptr_S]] %[[#]] %[[#ulong_2]]
; CHECK:	  %[[#B:]] = OpInBoundsAccessChain %[[#ptr_uint]] %[[#A]] %[[#ulong_1]]
; CHECK:	  %[[#C:]] = OpLoad %[[#uint]] %[[#B]] Aligned 4

  %4 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([5 x %struct.S]) %tmp, i64 3)
  %5 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.S) %4, i32 0)
  store i32 %3, ptr %5, align 4
; CHECK:	  %[[#D:]] = OpInBoundsAccessChain %[[#ptr_S]] %[[#tmp]] %[[#ulong_3]]
; CHECK:	  %[[#E:]] = OpInBoundsAccessChain %[[#ptr_uint]] %[[#D]] %[[#uint_0]]
; CHECK:	             OpStore %[[#E]] %[[#C]] Aligned 4

  ret void
}

define spir_func void @array_struct_load_store_combined(ptr %a) convergent {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %tmp = alloca [5 x %struct.S], align 4
; CHECK:	%[[#tmp:]] = OpVariable %[[#ptr_arr_S_5]] Function

  %1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([3 x %struct.S]) %a, i64 2, i64 1)
  %2 = load i32, ptr %1, align 4
; CHECK:	  %[[#A:]] = OpInBoundsAccessChain %[[#ptr_uint]] %[[#]] %[[#ulong_2]] %[[#ulong_1]]
; CHECK:	  %[[#B:]] = OpLoad %[[#uint]] %[[#A]] Aligned 4

  %3 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([5 x %struct.S]) %tmp, i64 3, i32 0)
  store i32 %2, ptr %3, align 4
; CHECK:	  %[[#C:]] = OpInBoundsAccessChain %[[#ptr_uint]] %[[#tmp]] %[[#ulong_3]] %[[#uint_0]]
; CHECK:	             OpStore %[[#C]] %[[#B]] Aligned 4

  ret void
}

define spir_func void @array_nested_struct_load_store(ptr %a) convergent {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %tmp = alloca [5 x %struct.S2], align 4
; CHECK:	%[[#tmp:]] = OpVariable %[[#ptr_arr_S2_5]] Function

  %1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([3 x %struct.S2]) %a, i64 1)
  %2 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.S2) %1, i64 1)
  %3 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.S) %2, i64 0)
  %4 = load i32, ptr %3, align 4
; CHECK:	  %[[#A:]] = OpInBoundsAccessChain %[[#ptr_S2]] %[[#]] %[[#ulong_1]]
; CHECK:	  %[[#B:]] = OpInBoundsAccessChain %[[#ptr_S]] %[[#A]] %[[#ulong_1]]
; CHECK:	  %[[#C:]] = OpInBoundsAccessChain %[[#ptr_uint]] %[[#B]] %[[#ulong_0]]
; CHECK:	  %[[#D:]] = OpLoad %[[#uint]] %[[#C]] Aligned 4

  %5 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([5 x %struct.S2]) %tmp, i64 3)
  %6 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.S2) %5, i32 1)
  %7 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.S) %6, i32 0)
  store i32 %4, ptr %7, align 4
; CHECK:	  %[[#E:]] = OpInBoundsAccessChain %[[#ptr_S2]] %[[#tmp]] %[[#ulong_3]]
; CHECK:	  %[[#F:]] = OpInBoundsAccessChain %[[#ptr_S]] %[[#E]] %[[#uint_1]]
; CHECK:	  %[[#G:]] = OpInBoundsAccessChain %[[#ptr_uint]] %[[#F]] %[[#uint_0]]
; CHECK:	             OpStore %[[#G]] %[[#D]] Aligned 4

  ret void
}

define spir_func void @array_nested_struct_load_store_combined(ptr %a) convergent {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %tmp = alloca [5 x %struct.S2], align 4
; CHECK:	%[[#tmp:]] = OpVariable %[[#ptr_arr_S2_5]] Function

  %1 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([3 x %struct.S2]) %a, i64 1, i64 1, i64 0)
  %2 = load i32, ptr %1, align 4
; CHECK:	  %[[#A:]] = OpInBoundsAccessChain %[[#ptr_uint]] %[[#]] %[[#ulong_1]] %[[#ulong_1]] %[[#ulong_0]]
; CHECK:	  %[[#B:]] = OpLoad %[[#uint]] %[[#A]] Aligned 4

  %3 = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([5 x %struct.S2]) %tmp, i64 3, i32 1, i32 0)
  store i32 %2, ptr %3, align 4
; CHECK:	  %[[#C:]] = OpInBoundsAccessChain %[[#ptr_uint]] %[[#tmp]] %[[#ulong_3]] %[[#uint_1]] %[[#uint_0]]
; CHECK:	             OpStore %[[#C]] %[[#B]] Aligned 4

  ret void
}

declare token @llvm.experimental.convergence.entry() #1

declare ptr @llvm.structured.gep.p0(ptr, ...) #3

attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
