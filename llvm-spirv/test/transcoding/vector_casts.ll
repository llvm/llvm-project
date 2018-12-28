; ModuleID = '/nfs/site/home/aelizuno/tmp/conversions.bc'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s

; Check vector conversions w\o decorations are translated back to calls
; to OCL built-ins.

; CHECK:    call spir_func <8 x i16> @_Z14convert_short8Dv8_c(<8 x i8>
; CHECK:    call spir_func <8 x i32> @_Z12convert_int8Dv8_t(<8 x i16>
; CHECK:    call spir_func <8 x i8> @_Z13convert_char8Dv8_i(<8 x i32>
; CHECK:    call spir_func <8 x double> @_Z15convert_double8Dv8_c(<8 x i8>
; CHECK:    call spir_func <8 x float> @_Z14convert_float8Dv8_d(<8 x double>
; CHECK:    call spir_func <8 x double> @_Z15convert_double8Dv8_f(<8 x float>
; CHECK:    call spir_func <8 x i32> @_Z13convert_uint8Dv8_d(<8 x double>
; CHECK:    call spir_func <8 x float> @_Z14convert_float8Dv8_j(<8 x i32>
; CHECK:    call spir_func <8 x i32> @_Z12convert_int8Dv8_f(<8 x float>

; Function Attrs: nounwind
define spir_kernel void @test_default_conversions(<8 x double> addrspace(1)* nocapture %out, <8 x i8> %in) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !5 !kernel_arg_type_qual !4 {
  %1 = tail call spir_func <8 x i16> @_Z15convert_ushort8Dv8_c(<8 x i8> %in) #1
  %2 = tail call spir_func <8 x i32> @_Z12convert_int8Dv8_t(<8 x i16> %1) #1
  %3 = tail call spir_func <8 x i8> @_Z13convert_char8Dv8_i(<8 x i32> %2) #1
  %4 = tail call spir_func <8 x double> @_Z15convert_double8Dv8_c(<8 x i8> %3) #1
  %5 = tail call spir_func <8 x float> @_Z14convert_float8Dv8_d(<8 x double> %4) #1
  %6 = tail call spir_func <8 x double> @_Z15convert_double8Dv8_f(<8 x float> %5) #1
  %7 = tail call spir_func <8 x i32> @_Z13convert_uint8Dv8_d(<8 x double> %6) #1
  %8 = tail call spir_func <8 x float> @_Z14convert_float8Dv8_j(<8 x i32> %7) #1
  %9 = tail call spir_func <8 x i32> @_Z12convert_int8Dv8_f(<8 x float> %8) #1
  %10 = tail call spir_func <8 x double> @_Z15convert_double8Dv8_i(<8 x i32> %9) #1
  store <8 x double> %10, <8 x double> addrspace(1)* %out, align 64, !tbaa !9
  ret void
}

; Function Attrs: nounwind readnone
declare spir_func <8 x i16> @_Z15convert_ushort8Dv8_c(<8 x i8>) #1

; Function Attrs: nounwind readnone
declare spir_func <8 x i32> @_Z12convert_int8Dv8_t(<8 x i16>) #1

; Function Attrs: nounwind readnone
declare spir_func <8 x i8> @_Z13convert_char8Dv8_i(<8 x i32>) #1

; Function Attrs: nounwind readnone
declare spir_func <8 x double> @_Z15convert_double8Dv8_c(<8 x i8>) #1

; Function Attrs: nounwind readnone
declare spir_func <8 x float> @_Z14convert_float8Dv8_d(<8 x double>) #1

; Function Attrs: nounwind readnone
declare spir_func <8 x double> @_Z15convert_double8Dv8_f(<8 x float>) #1

; Function Attrs: nounwind readnone
declare spir_func <8 x i32> @_Z13convert_uint8Dv8_d(<8 x double>) #1

; Function Attrs: nounwind readnone
declare spir_func <8 x float> @_Z14convert_float8Dv8_j(<8 x i32>) #1

; Function Attrs: nounwind readnone
declare spir_func <8 x i32> @_Z12convert_int8Dv8_f(<8 x float>) #1

; Function Attrs: nounwind readnone
declare spir_func <8 x double> @_Z15convert_double8Dv8_i(<8 x i32>) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!6}
!opencl.used.extensions = !{!7}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!7}

!1 = !{i32 1, i32 0}
!2 = !{!"none", !"none"}
!3 = !{!"double8*", !"char8"}
!4 = !{!"", !""}
!5 = !{!"double8*", !"char8"}
!6 = !{i32 1, i32 2}
!7 = !{}
!8 = !{!"cl_doubles"}
!9 = !{!10, !10, i64 0}
!10 = !{!"omnipotent char", !11}
!11 = !{!"Simple C/C++ TBAA"}
