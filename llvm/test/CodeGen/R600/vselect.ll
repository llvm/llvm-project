;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK: @test_select_v4i32
; CHECK: CNDE_INT T{{[0-9]+\.[XYZW], PV\.[xyzw], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: CNDE_INT * T{{[0-9]+\.[XYZW], PV\.[xyzw], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: CNDE_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: CNDE_INT * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define void @test_select_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in0, <4 x i32> addrspace(1)* %in1) {
entry:
  %0 = load <4 x i32> addrspace(1)* %in0
  %1 = load <4 x i32> addrspace(1)* %in1
  %cmp = icmp ne <4 x i32> %0, %1
  %result = select <4 x i1> %cmp, <4 x i32> %0, <4 x i32> %1
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}
