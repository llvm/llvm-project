; RUN: llc -march=r600 -mcpu=cypress < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; XFAIL: *

; This is the failing part of the r600 bitacts tests

; TODO: enable doubles
; FUNC-LABEL: {{^}}bitcast_f64_to_v2i32:
define amdgpu_kernel void @bitcast_f64_to_v2i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load double, ptr addrspace(1) %in, align 8
  %add = fadd double %val, 4.0
  %bc = bitcast double %add to <2 x i32>
  store <2 x i32> %bc, ptr addrspace(1) %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}bitcast_v2i64_to_v2f64:
define amdgpu_kernel void @bitcast_v2i64_to_v2f64(i32 %cond, ptr addrspace(1) %out, <2 x i64> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <2 x i64> %value to <2 x double>
  br label %end

end:
  %phi = phi <2 x double> [zeroinitializer, %entry], [%cast, %if]
  store <2 x double> %phi, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}bitcast_v2f64_to_v2i64:
define amdgpu_kernel void @bitcast_v2f64_to_v2i64(i32 %cond, ptr addrspace(1) %out, <2 x double> %value) {
entry:
  %cmp0 = icmp eq i32 %cond, 0
  br i1 %cmp0, label %if, label %end

if:
  %cast = bitcast <2 x double> %value to <2 x i64>
  br label %end

end:
  %phi = phi <2 x i64> [zeroinitializer, %entry], [%cast, %if]
  store <2 x i64> %phi, ptr addrspace(1) %out
  ret void
}
