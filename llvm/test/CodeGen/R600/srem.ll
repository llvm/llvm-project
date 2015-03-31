; RUN: llc -march=amdgcn -mcpu=SI < %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s
; RUN: llc -march=r600 -mcpu=redwood < %s

define void @srem_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %den_ptr = getelementptr i32 addrspace(1)* %in, i32 1
  %num = load i32 addrspace(1) * %in
  %den = load i32 addrspace(1) * %den_ptr
  %result = srem i32 %num, %den
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

define void @srem_i32_4(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %num = load i32 addrspace(1) * %in
  %result = srem i32 %num, 4
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

define void @srem_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %den_ptr = getelementptr <2 x i32> addrspace(1)* %in, i32 1
  %num = load <2 x i32> addrspace(1) * %in
  %den = load <2 x i32> addrspace(1) * %den_ptr
  %result = srem <2 x i32> %num, %den
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

define void @srem_v2i32_4(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %num = load <2 x i32> addrspace(1) * %in
  %result = srem <2 x i32> %num, <i32 4, i32 4>
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

define void @srem_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %den_ptr = getelementptr <4 x i32> addrspace(1)* %in, i32 1
  %num = load <4 x i32> addrspace(1) * %in
  %den = load <4 x i32> addrspace(1) * %den_ptr
  %result = srem <4 x i32> %num, %den
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

define void @srem_v4i32_4(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %num = load <4 x i32> addrspace(1) * %in
  %result = srem <4 x i32> %num, <i32 4, i32 4, i32 4, i32 4>
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}
