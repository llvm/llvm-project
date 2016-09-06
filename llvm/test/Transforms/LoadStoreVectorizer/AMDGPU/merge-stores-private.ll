; RUN: opt -mtriple=amdgcn-amd-amdhsa -mattr=+max-private-element-size-4 -load-store-vectorizer -S -o - %s | FileCheck -check-prefix=ELT4 -check-prefix=ALL %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mattr=+max-private-element-size-8 -load-store-vectorizer -S -o - %s | FileCheck -check-prefix=ELT8 -check-prefix=ALL %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mattr=+max-private-element-size-16 -load-store-vectorizer -S -o - %s | FileCheck -check-prefix=ELT16 -check-prefix=ALL %s

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-p24:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"

; ALL-LABEL: @merge_private_store_4_vector_elts_loads_v4i32
; ELT4: store i32
; ELT4: store i32
; ELT4: store i32
; ELT4: store i32

; ELT8: store <2 x i32>
; ELT8: store <2 x i32>

; ELT16: store <4 x i32>
define void @merge_private_store_4_vector_elts_loads_v4i32(i32* %out) #0 {
  %out.gep.1 = getelementptr i32, i32* %out, i32 1
  %out.gep.2 = getelementptr i32, i32* %out, i32 2
  %out.gep.3 = getelementptr i32, i32* %out, i32 3

  store i32 9, i32* %out
  store i32 1, i32* %out.gep.1
  store i32 23, i32* %out.gep.2
  store i32 19, i32* %out.gep.3
  ret void
}

; ALL-LABEL: @merge_private_store_4_vector_elts_loads_v4i8(
; ALL: store <4 x i8>
define void @merge_private_store_4_vector_elts_loads_v4i8(i8* %out) #0 {
  %out.gep.1 = getelementptr i8, i8* %out, i32 1
  %out.gep.2 = getelementptr i8, i8* %out, i32 2
  %out.gep.3 = getelementptr i8, i8* %out, i32 3

  store i8 9, i8* %out, align 4
  store i8 1, i8* %out.gep.1
  store i8 23, i8* %out.gep.2
  store i8 19, i8* %out.gep.3
  ret void
}

; ALL-LABEL: @merge_private_store_4_vector_elts_loads_v2i16(
; ALL: store <2 x i16>
define void @merge_private_store_4_vector_elts_loads_v2i16(i16* %out) #0 {
  %out.gep.1 = getelementptr i16, i16* %out, i32 1

  store i16 9, i16* %out, align 4
  store i16 12, i16* %out.gep.1
  ret void
}

attributes #0 = { nounwind }
