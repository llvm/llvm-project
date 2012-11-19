; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s --check-prefix=PTX32
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=PTX64

define ptx_kernel void @t1(i1* %a) {
; PTX32:      mov.u16 %rc{{[0-9]+}}, 0;
; PTX32-NEXT: st.u8 [%r{{[0-9]+}}], %rc{{[0-9]+}};
; PTX64:      mov.u16 %rc{{[0-9]+}}, 0;
; PTX64-NEXT: st.u8 [%rl{{[0-9]+}}], %rc{{[0-9]+}};
  store i1 false, i1* %a
  ret void
}


define ptx_kernel void @t2(i1* %a, i8* %b) {
; PTX32: ld.u8 %rc{{[0-9]+}}, [%r{{[0-9]+}}]
; PTX32: and.b16 temp, %rc{{[0-9]+}}, 1;
; PTX32: setp.b16.eq %p{{[0-9]+}}, temp, 1;
; PTX64: ld.u8 %rc{{[0-9]+}}, [%rl{{[0-9]+}}]
; PTX64: and.b16 temp, %rc{{[0-9]+}}, 1;
; PTX64: setp.b16.eq %p{{[0-9]+}}, temp, 1;

  %t1 = load i1* %a
  %t2 = select i1 %t1, i8 1, i8 2
  store i8 %t2, i8* %b
  ret void
}
