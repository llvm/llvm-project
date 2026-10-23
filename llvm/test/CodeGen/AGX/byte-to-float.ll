; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; The AGX u8 conversion reads only the low byte. Fold the mask before a
; zero-extended byte-to-float conversion, including bytes from vector loads.
; CHECK-LABEL: masked_u8:
; CHECK-NOT: and {{.*}}, 255
; CHECK: convert u8_to_f
define agx_kernel void @masked_u8(ptr addrspace(1) %out, i16 %x) {
  %byte = and i16 %x, 255
  %f = uitofp i16 %byte to float
  store float %f, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: vector_bytes:
; CHECK: device_load {{[01]}}, i8, xyzw
; CHECK-NOT: and {{.*}}, 255
; CHECK: convert u8_to_f
; CHECK: convert u8_to_f
; CHECK: convert u8_to_f
; CHECK: convert u8_to_f
define agx_kernel void @vector_bytes(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %v = load <4 x i8>, ptr addrspace(1) %in, align 4
  %a = extractelement <4 x i8> %v, i32 0
  %b = extractelement <4 x i8> %v, i32 1
  %c = extractelement <4 x i8> %v, i32 2
  %d = extractelement <4 x i8> %v, i32 3
  %a16 = zext i8 %a to i16
  %b16 = zext i8 %b to i16
  %c16 = zext i8 %c to i16
  %d16 = zext i8 %d to i16
  %af = uitofp i16 %a16 to float
  %bf = uitofp i16 %b16 to float
  %cf = uitofp i16 %c16 to float
  %df = uitofp i16 %d16 to float
  %o1 = getelementptr float, ptr addrspace(1) %out, i32 1
  %o2 = getelementptr float, ptr addrspace(1) %out, i32 2
  %o3 = getelementptr float, ptr addrspace(1) %out, i32 3
  store float %af, ptr addrspace(1) %out
  store float %bf, ptr addrspace(1) %o1
  store float %cf, ptr addrspace(1) %o2
  store float %df, ptr addrspace(1) %o3
  ret void
}

; CHECK-LABEL: full_u16:
; CHECK: convert u16_to_f
define agx_kernel void @full_u16(ptr addrspace(1) %out, i16 %x) {
  %f = uitofp i16 %x to float
  store float %f, ptr addrspace(1) %out
  ret void
}
