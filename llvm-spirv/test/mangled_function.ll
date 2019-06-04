; Make sure translator doesn't crash if the program contains
; a mangled function which is not an OpenCL bultin.
; Source:
; void __attribute__((overloadable))
; foo(image2d_t srcImage);
;
; void bar(image2d_t srcImage) {
;   foo(srcImage);
; }
; clang -cc1 /work/tmp/tmp.cl -cl-std=CL2.0 -triple spir-unknown-unknown  -finclude-default-header -emit-llvm -o test/mangled_function.ll

; RUN: llvm-as < %s | llvm-spirv -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -o - -to-text | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o - -r | llvm-dis | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Name [[foo:[0-9]+]] "_Z3foo14ocl_image2d_ro"
; CHECK-SPIRV: Function {{[0-9]+}} [[foo]]

; ModuleID = '/work/tmp/tmp.cl'
source_filename = "/work/tmp/tmp.cl"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

%opencl.image2d_ro_t = type opaque

; Function Attrs: convergent nounwind
define spir_func void @bar(%opencl.image2d_ro_t addrspace(1)* %srcImage) local_unnamed_addr #0 {
entry:
; CHECK-LLVM: call spir_func void @_Z3foo14ocl_image2d_ro(%opencl.image2d_ro_t addrspace(1)* %srcImage)
  tail call spir_func void @_Z3foo14ocl_image2d_ro(%opencl.image2d_ro_t addrspace(1)* %srcImage) #2
  ret void
}

; Function Attrs: convergent
declare spir_func void @_Z3foo14ocl_image2d_ro(%opencl.image2d_ro_t addrspace(1)*) local_unnamed_addr #1

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 9.0.0 (https://llvm.org/git/clang d81772e8081d5af6925826ebf18ade9dd0323bb9) (https://llvm.org/git/llvm 5a295517ec57c58837c5fe5cf364c0f2e609865f)"}
