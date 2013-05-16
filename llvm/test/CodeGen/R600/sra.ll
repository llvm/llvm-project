; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK: @ashr_v4i32
; CHECK: ASHR * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: ASHR * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: ASHR * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; CHECK: ASHR * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define void @ashr_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> %a, <4 x i32> %b) {
  %result = ashr <4 x i32> %a, %b
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}
