; Verify register types we generate in PTX.
; RUN: llc -O0 < %s -mtriple=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc -O0 < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: llc -O0 < %s -mtriple=nvptx -mcpu=sm_20 | FileCheck %s -check-prefixes=NO8BIT
; RUN: llc -O0 < %s -mtriple=nvptx64 -mcpu=sm_20 | FileCheck %s -check-prefixes=NO8BIT
; RUN: %if ptxas-ptr32 %{ llc -O0 < %s -mtriple=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc -O0 < %s -mtriple=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; CHECK-LABEL: .visible .func func(
; NO8BIT-LABEL: .visible .func func(
define void @func(ptr %p, i1 %cond) {
; Both 8- and 16-bit integers are packed into 16-bit registers. So we should 
; not generate 8-bit registers.
; NO8BIT-NOT: .reg .{{[bsu]}}8

; CHECK-DAG: .reg .pred %p<
; CHECK-DAG: .reg .b16 %rs<
; CHECK-DAG: .reg .b32 %r<
; CHECK-DAG: .reg .b64 %rd<

entry:
  br i1 %cond, label %if, label %join
if:
  br label %join
join:
  ; CHECK-DAG: mov.pred %p{{[0-9]+}}, %p{{[0-9]+}};
  ; CHECK-DAG: mov.b16 %rs{{[0-9]+}}, %rs{{[0-9]+}};
  ; CHECK-DAG: mov.b16 %rs{{[0-9]+}}, %rs{{[0-9]+}};
  ; CHECK-DAG: mov.b32 %r{{[0-9]+}}, %r{{[0-9]+}};
  ; CHECK-DAG: mov.b64 %rd{{[0-9]+}}, %rd{{[0-9]+}};

  ; CHECK-DAG: mov.b32 %r{{[0-9]+}}, %r{{[0-9]+}};
  ; CHECK-DAG: mov.b64 %rd{{[0-9]+}}, %rd{{[0-9]+}};

  %v1 = phi i1 [ true, %if ], [ false, %entry ]
  %v8 = phi i8 [ 1, %if ], [ 0, %entry ]
  %v16 = phi i16 [ 2, %if ], [ 0, %entry ]
  %v32 = phi i32 [ 3, %if ], [ 0, %entry ]
  %v64 = phi i64 [ 4, %if ], [ 0, %entry ]
  %f32 = phi float [ 5.0, %if ], [ 0.0, %entry ]
  %f64 = phi double [ 6.0, %if ], [ 0.0, %entry ]

  store volatile i1 %v1, ptr %p, align 1
  store volatile i8 %v8, ptr %p, align 1
  store volatile i16 %v16, ptr %p, align 2
  store volatile i32 %v32, ptr %p, align 4
  store volatile i64 %v64, ptr %p, align 8
  store volatile float %f32, ptr %p, align 4
  store volatile double %f64, ptr %p, align 8
  ret void
}

