;; struct S1
;; {
;;   int x;
;;   int y;
;;   int z;
;; };
;; S1 foo11()
;; {
;;   return S1();
;; }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: TypeInt [[Int8:[0-9]+]] 8 0
; CHECK-SPIRV: Constant {{[0-9]+}} [[Len:[0-9]+]] 12
; CHECK-SPIRV: TypePointer [[Int8Ptr:[0-9]+]] 8 [[Int8]]
; CHECK-SPIRV: TypeArray [[Int8x12:[0-9]+]] [[Int8]] [[Len]]
; CHECK-SPIRV: TypePointer [[Int8PtrConst:[0-9]+]] 0 [[Int8]]
; CHECK-SPIRV: ConstantNull [[Int8x12]] [[Init:[0-9]+]]
; CHECK-SPIRV: Variable {{[0-9]+}} [[Val:[0-9]+]] 0 [[Init]]

; CHECK-SPIRV: Bitcast [[Int8Ptr]] [[Target:[0-9]+]] {{[0-9]+}}
; CHECK-SPIRV: Bitcast [[Int8PtrConst]] [[Source:[0-9]+]] [[Val]]
; CHECK-SPIRV: CopyMemorySized [[Target]] [[Source]] [[Len]] 2 4


target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir"

%struct.S1 = type { i32, i32, i32 }

; Function Attrs: nounwind
define spir_func void @_Z5foo11v(%struct.S1 addrspace(4)* noalias nocapture sret %agg.result) #0 {
  %1 = bitcast %struct.S1 addrspace(4)* %agg.result to i8 addrspace(4)*
  tail call void @llvm.memset.p4i8.i32(i8 addrspace(4)* %1, i8 0, i32 12, i32 4, i1 false)
; CHECK-LLVM: call void @llvm.memset.p4i8.i32(i8 addrspace(4)* align 4 %1, i8 0, i32 12, i1 false)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memset.p4i8.i32(i8 addrspace(4)* nocapture, i8, i32, i32, i1) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!3}
!spirv.Source = !{!4}
!spirv.String = !{!5}

!0 = !{i32 1, i32 2}
!1 = !{i32 2, i32 2}
!2 = !{}
!3 = !{!"clang version 3.6.1 "}
!4 = !{i32 4, i32 202000, !5}
!5 = !{!"llvm.memset.cl"}
