; RUN: opt -passes=gvn -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-p100:128:64:64-p101:128:64:64"
target triple = "x86_64-unknown-linux-gnu"

%ArrayImpl = type { i64, i64 addrspace(100)*, [1 x i64], [1 x i64], [1 x i64], i64, i64, double addrspace(100)*, double addrspace(100)*, i8, i64 }
%_array = type { i64, %ArrayImpl addrspace(100)*, i8 }

define void @test(i64 %n_chpl) {
entry:
  ; First section is some code
  %0 = getelementptr inbounds %_array, %_array* null, i32 0, i32 1
  %1 = load %ArrayImpl addrspace(100)*, %ArrayImpl addrspace(100)** %0
  %2 = getelementptr inbounds %ArrayImpl, %ArrayImpl addrspace(100)* %1, i32 0, i32 8
  %3 = load double addrspace(100)*, double addrspace(100)* addrspace(100)* %2
  %4 = getelementptr inbounds double, double addrspace(100)* %3, i64 -1
  ; Second section is that code repeated
  %x0 = getelementptr inbounds %_array, %_array* null, i32 0, i32 1
  %x1 = load %ArrayImpl addrspace(100)*, %ArrayImpl addrspace(100)** %x0
  %x2 = getelementptr inbounds %ArrayImpl, %ArrayImpl addrspace(100)* %x1, i32 0, i32 8
  %x3 = load double addrspace(100)*, double addrspace(100)* addrspace(100)* %x2
  %x4 = getelementptr inbounds double, double addrspace(100)* %x3, i64 -1
  ; These two stores refer to the same memory location
  ; Even so, they are expected to remain separate stores here
  store double 0.000000e+00, double addrspace(100)* %4
  store double 0.000000e+00, double addrspace(100)* %x4
  ; Third section is the repeated code again, with a later store
  ; This third section is necessary to trigger the crash
  %y1 = load %ArrayImpl addrspace(100)*, %ArrayImpl addrspace(100)** %0
  %y2 = getelementptr inbounds %ArrayImpl, %ArrayImpl addrspace(100)* %y1, i32 0, i32 8
  %y3 = load double addrspace(100)*, double addrspace(100)* addrspace(100)* %y2
  %y4 = getelementptr inbounds double, double addrspace(100)* %y3, i64 -1
  store double 0.000000e+00, double addrspace(100)* %y4
  ret void
; CHECK-LABEL: define void @test
; CHECK: getelementptr inbounds double, double addrspace(100)* {{%.*}}, i64 -1
; CHECK-NEXT: store double 0.000000e+00, double addrspace(100)* [[DST:%.*]]
; CHECK-NEXT: store double 0.000000e+00, double addrspace(100)* [[DST]]
; CHECK: load
; CHECK: getelementptr inbounds %ArrayImpl, %ArrayImpl addrspace(100)*
; CHECK: load
; CHECK: getelementptr inbounds double, double addrspace(100)* {{%.*}}, i64 -1
; CHECK: store double 0.000000e+00, double addrspace(100)*
; CHECK: ret
}
