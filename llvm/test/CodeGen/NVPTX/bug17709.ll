; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}

; ModuleID = '__kernelgen_main_module'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define private ptx_device { double, double } @__utils1_MOD_trace(ptr noalias %m) {
entry:
  ;unreachable
  %t0 = insertvalue {double, double} undef, double 1.0, 0
  %t1 = insertvalue {double, double} %t0, double 1.0, 1
  ret { double, double } %t1
}

%struct.descriptor_dimension.0.52 = type { i64, i64, i64 }
%"struct.array2_complex(kind=8).37.18.70" = type { ptr, i64, i64, [2 x %struct.descriptor_dimension.0.52] }
%"struct.array2_complex(kind=8).43.5.57" = type { ptr, i64, i64, [2 x %struct.descriptor_dimension.0.52] }
@replacementOfAlloca8 = private global %"struct.array2_complex(kind=8).37.18.70" zeroinitializer, align 4096

; CHECK: .visible .entry __kernelgen_main
define ptx_kernel void @__kernelgen_main(ptr nocapture %args, ptr) {
entry:
  %1 = tail call ptx_device { double, double } @__utils1_MOD_trace(ptr noalias @replacementOfAlloca8)
  ret void
}

