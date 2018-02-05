; RUN: opt -data-layout=A5 -S -load-store-vectorizer -mattr=-unaligned-buffer-access,+max-private-element-size-16 < %s | FileCheck -check-prefix=ALIGNED -check-prefix=ALL %s
; RUN: opt -data-layout=A5 -S -load-store-vectorizer -mattr=+unaligned-buffer-access,+unaligned-scratch-access,+max-private-element-size-16 < %s | FileCheck -check-prefix=UNALIGNED -check-prefix=ALL %s

target triple = "amdgcn--"

; ALL-LABEL: @load_unknown_offset_align1_i8(
; ALL: alloca [128 x i8], align 1
; UNALIGNED: load <2 x i8>, <2 x i8> addrspace(5)* %{{[0-9]+}}, align 1{{$}}

; ALIGNED: load i8, i8 addrspace(5)* %ptr0, align 1{{$}}
; ALIGNED: load i8, i8 addrspace(5)* %ptr1, align 1{{$}}
define amdgpu_kernel void @load_unknown_offset_align1_i8(i8 addrspace(1)* noalias %out, i32 %offset) #0 {
  %alloca = alloca [128 x i8], align 1, addrspace(5)
  %ptr0 = getelementptr inbounds [128 x i8], [128 x i8] addrspace(5)* %alloca, i32 0, i32 %offset
  %val0 = load i8, i8 addrspace(5)* %ptr0, align 1
  %ptr1 = getelementptr inbounds i8, i8 addrspace(5)* %ptr0, i32 1
  %val1 = load i8, i8 addrspace(5)* %ptr1, align 1
  %add = add i8 %val0, %val1
  store i8 %add, i8 addrspace(1)* %out
  ret void
}

; ALL-LABEL: @load_unknown_offset_align1_i16(
; ALL: alloca [128 x i16], align 1, addrspace(5){{$}}
; UNALIGNED: load <2 x i16>, <2 x i16> addrspace(5)* %{{[0-9]+}}, align 1{{$}}

; ALIGNED: load i16, i16 addrspace(5)* %ptr0, align 1{{$}}
; ALIGNED: load i16, i16 addrspace(5)* %ptr1, align 1{{$}}
define amdgpu_kernel void @load_unknown_offset_align1_i16(i16 addrspace(1)* noalias %out, i32 %offset) #0 {
  %alloca = alloca [128 x i16], align 1, addrspace(5)
  %ptr0 = getelementptr inbounds [128 x i16], [128 x i16] addrspace(5)* %alloca, i32 0, i32 %offset
  %val0 = load i16, i16 addrspace(5)* %ptr0, align 1
  %ptr1 = getelementptr inbounds i16, i16 addrspace(5)* %ptr0, i32 1
  %val1 = load i16, i16 addrspace(5)* %ptr1, align 1
  %add = add i16 %val0, %val1
  store i16 %add, i16 addrspace(1)* %out
  ret void
}

; FIXME: Although the offset is unknown here, we know it is a multiple
; of the element size, so should still be align 4

; ALL-LABEL: @load_unknown_offset_align1_i32(
; ALL: alloca [128 x i32], align 1
; UNALIGNED: load <2 x i32>, <2 x i32> addrspace(5)* %{{[0-9]+}}, align 1{{$}}

; ALIGNED: load i32, i32 addrspace(5)* %ptr0, align 1
; ALIGNED: load i32, i32 addrspace(5)* %ptr1, align 1
define amdgpu_kernel void @load_unknown_offset_align1_i32(i32 addrspace(1)* noalias %out, i32 %offset) #0 {
  %alloca = alloca [128 x i32], align 1, addrspace(5)
  %ptr0 = getelementptr inbounds [128 x i32], [128 x i32] addrspace(5)* %alloca, i32 0, i32 %offset
  %val0 = load i32, i32 addrspace(5)* %ptr0, align 1
  %ptr1 = getelementptr inbounds i32, i32 addrspace(5)* %ptr0, i32 1
  %val1 = load i32, i32 addrspace(5)* %ptr1, align 1
  %add = add i32 %val0, %val1
  store i32 %add, i32 addrspace(1)* %out
  ret void
}

; FIXME: Should always increase alignment of the load
; Make sure alloca alignment isn't decreased
; ALL-LABEL: @load_alloca16_unknown_offset_align1_i32(
; ALL: alloca [128 x i32], align 16

; UNALIGNED: load <2 x i32>, <2 x i32> addrspace(5)* %{{[0-9]+}}, align 1{{$}}

; FIXME: Should change alignment
; ALIGNED: load i32
; ALIGNED: load i32
define amdgpu_kernel void @load_alloca16_unknown_offset_align1_i32(i32 addrspace(1)* noalias %out, i32 %offset) #0 {
  %alloca = alloca [128 x i32], align 16, addrspace(5)
  %ptr0 = getelementptr inbounds [128 x i32], [128 x i32] addrspace(5)* %alloca, i32 0, i32 %offset
  %val0 = load i32, i32 addrspace(5)* %ptr0, align 1
  %ptr1 = getelementptr inbounds i32, i32 addrspace(5)* %ptr0, i32 1
  %val1 = load i32, i32 addrspace(5)* %ptr1, align 1
  %add = add i32 %val0, %val1
  store i32 %add, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL: @store_unknown_offset_align1_i8(
; ALL: alloca [128 x i8], align 1
; UNALIGNED: store <2 x i8> <i8 9, i8 10>, <2 x i8> addrspace(5)* %{{[0-9]+}}, align 1{{$}}

; ALIGNED: store i8 9, i8 addrspace(5)* %ptr0, align 1{{$}}
; ALIGNED: store i8 10, i8 addrspace(5)* %ptr1, align 1{{$}}
define amdgpu_kernel void @store_unknown_offset_align1_i8(i8 addrspace(1)* noalias %out, i32 %offset) #0 {
  %alloca = alloca [128 x i8], align 1, addrspace(5)
  %ptr0 = getelementptr inbounds [128 x i8], [128 x i8] addrspace(5)* %alloca, i32 0, i32 %offset
  store i8 9, i8 addrspace(5)* %ptr0, align 1
  %ptr1 = getelementptr inbounds i8, i8 addrspace(5)* %ptr0, i32 1
  store i8 10, i8 addrspace(5)* %ptr1, align 1
  ret void
}

; ALL-LABEL: @store_unknown_offset_align1_i16(
; ALL: alloca [128 x i16], align 1
; UNALIGNED: store <2 x i16> <i16 9, i16 10>, <2 x i16> addrspace(5)* %{{[0-9]+}}, align 1{{$}}

; ALIGNED: store i16 9, i16 addrspace(5)* %ptr0, align 1{{$}}
; ALIGNED: store i16 10, i16 addrspace(5)* %ptr1, align 1{{$}}
define amdgpu_kernel void @store_unknown_offset_align1_i16(i16 addrspace(1)* noalias %out, i32 %offset) #0 {
  %alloca = alloca [128 x i16], align 1, addrspace(5)
  %ptr0 = getelementptr inbounds [128 x i16], [128 x i16] addrspace(5)* %alloca, i32 0, i32 %offset
  store i16 9, i16 addrspace(5)* %ptr0, align 1
  %ptr1 = getelementptr inbounds i16, i16 addrspace(5)* %ptr0, i32 1
  store i16 10, i16 addrspace(5)* %ptr1, align 1
  ret void
}

; FIXME: Although the offset is unknown here, we know it is a multiple
; of the element size, so it still should be align 4.

; ALL-LABEL: @store_unknown_offset_align1_i32(
; ALL: alloca [128 x i32], align 1

; UNALIGNED: store <2 x i32> <i32 9, i32 10>, <2 x i32> addrspace(5)* %{{[0-9]+}}, align 1{{$}}

; ALIGNED: store i32 9, i32 addrspace(5)* %ptr0, align 1
; ALIGNED: store i32 10, i32 addrspace(5)* %ptr1, align 1
define amdgpu_kernel void @store_unknown_offset_align1_i32(i32 addrspace(1)* noalias %out, i32 %offset) #0 {
  %alloca = alloca [128 x i32], align 1, addrspace(5)
  %ptr0 = getelementptr inbounds [128 x i32], [128 x i32] addrspace(5)* %alloca, i32 0, i32 %offset
  store i32 9, i32 addrspace(5)* %ptr0, align 1
  %ptr1 = getelementptr inbounds i32, i32 addrspace(5)* %ptr0, i32 1
  store i32 10, i32 addrspace(5)* %ptr1, align 1
  ret void
}

attributes #0 = { nounwind }

