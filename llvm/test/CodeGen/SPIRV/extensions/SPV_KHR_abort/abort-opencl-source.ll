; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - -filetype=obj | spirv-val %}

;; This test models the actual LLVM IR that Clang produces for OpenCL C calls
;; to `__spirv_AbortKHR`, including the trailing `ret void` (the OpenCL ABI
;; emits the call as a regular spir_func call followed by a return). The
;; backend must drop the trailing return because `OpAbortKHR` is itself a
;; SPIR-V function-termination instruction; the resulting block must have no
;; instructions after `OpAbortKHR`.
;;
;; Source (compiled with: clang -cc1 -triple spir64-unknown-unknown
;;        -cl-std=CL2.0 -finclude-default-header -emit-llvm -O0):
;;
;;   void __spirv_AbortKHR(uint);
;;   __kernel void k_scalar(uint x) { __spirv_AbortKHR(x); }
;;
;;   typedef struct { uint a; uint b; uint c; } Msg;
;;   void __spirv_AbortKHR(Msg);
;;   __kernel void k_struct(uint a, uint b, uint c) {
;;       Msg m = { a, b, c };
;;       __spirv_AbortKHR(m);
;;   }

; CHECK-DAG: OpCapability AbortKHR
; CHECK-DAG: OpExtension "SPV_KHR_abort"

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#STRUCT:]] = OpTypeStruct %[[#I32]] %[[#I32]] %[[#I32]]

; Scalar argument: passed by value, lowered to OpAbortKHR with the i32 type.
; The trailing OpReturn from the OpenCL ABI must be dropped.
; CHECK:     OpAbortKHR %[[#I32]] %{{[0-9]+}}
; CHECK-NEXT: OpFunctionEnd

; Struct argument: passed by `byval` pointer, must be loaded so OpAbortKHR
; receives the composite by value. No OpReturn / OpUnreachable after it.
; CHECK:     %[[#LOADED:]] = OpLoad %[[#STRUCT]]
; CHECK:     OpAbortKHR %[[#STRUCT]] %[[#LOADED]]
; CHECK-NEXT: OpFunctionEnd

; CHECK-NOT: OpReturn{{[[:space:]]+}}OpFunctionEnd
; CHECK-NOT: OpUnreachable

%struct.Msg = type { i32, i32, i32 }

declare spir_func void @_Z16__spirv_AbortKHRj(i32) #0
declare spir_func void @_Z16__spirv_AbortKHR3Msg(ptr byval(%struct.Msg)) #0

define spir_kernel void @k_scalar(i32 noundef %x) {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  call spir_func void @_Z16__spirv_AbortKHRj(i32 noundef %0)
  ret void
}

define spir_kernel void @k_struct(i32 noundef %a, i32 noundef %b, i32 noundef %c) {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %c.addr = alloca i32, align 4
  %m = alloca %struct.Msg, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  store i32 %c, ptr %c.addr, align 4
  %p0 = getelementptr inbounds %struct.Msg, ptr %m, i32 0, i32 0
  %0 = load i32, ptr %a.addr, align 4
  store i32 %0, ptr %p0, align 4
  %p1 = getelementptr inbounds %struct.Msg, ptr %m, i32 0, i32 1
  %1 = load i32, ptr %b.addr, align 4
  store i32 %1, ptr %p1, align 4
  %p2 = getelementptr inbounds %struct.Msg, ptr %m, i32 0, i32 2
  %2 = load i32, ptr %c.addr, align 4
  store i32 %2, ptr %p2, align 4
  call spir_func void @_Z16__spirv_AbortKHR3Msg(ptr noundef byval(%struct.Msg) align 4 %m)
  ret void
}

attributes #0 = { convergent nounwind }
