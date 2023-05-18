; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=PTX64
; RUN: %if ptxas && !ptxas-12.0 %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

define ptx_kernel void @t1(ptr %a) {
; PTX32:      mov.u16 %rs{{[0-9]+}}, 0;
; PTX32-NEXT: st.global.u8 [%r{{[0-9]+}}], %rs{{[0-9]+}};
; PTX64:      mov.u16 %rs{{[0-9]+}}, 0;
; PTX64-NEXT: st.global.u8 [%rd{{[0-9]+}}], %rs{{[0-9]+}};
  store i1 false, ptr %a
  ret void
}


define ptx_kernel void @t2(ptr %a, ptr %b) {
; PTX32: ld.global.u8 %rs{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: and.b16 %rs{{[0-9]+}}, %rs{{[0-9]+}}, 1;
; PTX32: setp.eq.b16 %p{{[0-9]+}}, %rs{{[0-9]+}}, 1;
; PTX64: ld.global.u8 %rs{{[0-9]+}}, [%rd{{[0-9]+}}]
; PTX64: and.b16 %rs{{[0-9]+}}, %rs{{[0-9]+}}, 1;
; PTX64: setp.eq.b16 %p{{[0-9]+}}, %rs{{[0-9]+}}, 1;

  %t1 = load i1, ptr %a
  %t2 = select i1 %t1, i8 1, i8 2
  store i8 %t2, ptr %b
  ret void
}
