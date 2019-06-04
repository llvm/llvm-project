; RUN: llvm-as <%s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t_4mspirv.bc
; RUN: llvm-dis %t_4mspirv.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM
;
; CHECK-SPIRV-DAG: TypeInt [[i32:[0-9]+]] 32 0
; CHECK-SPIRV-DAG: TypeInt [[i8:[0-9]+]] 8 0
; CHECK-SPIRV-DAG: Constant [[i32]] [[one:[0-9]+]] 1
; CHECK-SPIRV-DAG: Constant [[i32]] [[two:[0-9]+]] 2
; CHECK-SPIRV-DAG: Constant [[i32]] [[three:[0-9]+]] 3
; CHECK-SPIRV-DAG: Constant [[i32]] [[twelve:[0-9]+]] 12
; CHECK-SPIRV-DAG: TypeArray [[i32x3:[0-9]+]] [[i32]] [[three]]
; CHECK-SPIRV-DAG: TypePointer [[i32x3_ptr:[0-9]+]] 7 [[i32x3]]
; CHECK-SPIRV-DAG: TypePointer [[const_i32x3_ptr:[0-9]+]] 0 [[i32x3]]
; CHECK-SPIRV-DAG: TypePointer [[i8_ptr:[0-9]+]] 7 [[i8]]
; CHECK-SPIRV-DAG: TypePointer [[const_i8_ptr:[0-9]+]] 0 [[i8]]
; CHECK-SPIRV: ConstantComposite [[i32x3]] [[test_arr_init:[0-9]+]] [[one]] [[two]] [[three]]
; CHECK-SPIRV: Variable [[const_i32x3_ptr]] [[test_arr:[0-9]+]] 0 [[test_arr_init]]
; CHECK-SPIRV: Variable [[const_i32x3_ptr]] [[test_arr2:[0-9]+]] 0 [[test_arr_init]]
;
; CHECK-SPIRV: Variable [[i32x3_ptr]] [[arr:[0-9]+]] 7
; CHECK-SPIRV: Variable [[i32x3_ptr]] [[arr2:[0-9]+]] 7
;
; CHECK-SPIRV: Bitcast [[i8_ptr]] [[arr_i8_ptr:[0-9]+]] [[arr]]
; CHECK-SPIRV: Bitcast [[const_i8_ptr]] [[test_arr_const_i8_ptr:[0-9]+]] [[test_arr]]
; CHECK-SPIRV: CopyMemorySized [[arr_i8_ptr]] [[test_arr_const_i8_ptr]] [[twelve]] 2 4
;
; CHECK-SPIRV: Bitcast [[i8_ptr]] [[arr2_i8_ptr:[0-9]+]] [[arr2]]
; CHECK-SPIRV: Bitcast [[const_i8_ptr]] [[test_arr2_const_i8_ptr:[0-9]+]] [[test_arr2]]
; CHECK-SPIRV: CopyMemorySized [[arr2_i8_ptr]] [[test_arr2_const_i8_ptr]] [[twelve]] 2 4

; CHECK-LLVM:@__const.test.arr = internal addrspace(2) constant [3 x i32] [i32 1, i32 2, i32 3], align 4 
; CHECK-LLVM-DAG:@__const.test.arr2 = internal addrspace(2) constant [3 x i32] [i32 1, i32 2, i32 3], align 4 

; CHECK-LLVM:%[[VAR1:.*]] = bitcast [3 x i32] addrspace(2)* @__const.test.arr to i8 addrspace(2)* 
; CHECK-LLVM-DAG:call void @llvm.memcpy.p0i8.p2i8.i32(i8* align 4 %[[VAR0:.*]], i8 addrspace(2)* align 4 %[[VAR1]], i32 12, i1 false)

; CHECK-LLVM:%[[VAR3:.*]] = bitcast [3 x i32] addrspace(2)* @__const.test.arr2 to i8 addrspace(2)* 
; CHECK-LLVM-DAG:call void @llvm.memcpy.p0i8.p2i8.i32(i8* align 4 %[[VAR2:.*]], i8 addrspace(2)* align 4 %[[VAR3]], i32 12, i1 false)

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

@__const.test.arr = private unnamed_addr addrspace(2) constant [3 x i32] [i32 1, i32 2, i32 3], align 4
@__const.test.arr2 = private unnamed_addr addrspace(2) constant [3 x i32] [i32 1, i32 2, i32 3], align 4

; Function Attrs: convergent noinline nounwind optnone
define spir_func void @test() #0 {
entry:
  %arr = alloca [3 x i32], align 4
  %arr2 = alloca [3 x i32], align 4
  %0 = bitcast [3 x i32]* %arr to i8*
  call void @llvm.memcpy.p0i8.p2i8.i32(i8* align 4 %0, i8 addrspace(2)* align 4 bitcast ([3 x i32] addrspace(2)* @__const.test.arr to i8 addrspace(2)*), i32 12, i1 false)
  %1 = bitcast [3 x i32]* %arr2 to i8*
  call void @llvm.memcpy.p0i8.p2i8.i32(i8* align 4 %1, i8 addrspace(2)* align 4 bitcast ([3 x i32] addrspace(2)* @__const.test.arr2 to i8 addrspace(2)*), i32 12, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p2i8.i32(i8* nocapture writeonly, i8 addrspace(2)* nocapture readonly, i32, i1) #1

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
