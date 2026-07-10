; RUN: opt < %s -passes=slp-vectorizer -S -mtriple=nvptx64-nvidia-cuda -mcpu=sm_100 | FileCheck %s

target triple = "nvptx64-nvidia-cuda"

define void @pair8_copy(ptr addrspace(1) %in, ptr addrspace(1) %out) {
; CHECK-LABEL: @pair8_copy(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i32>, ptr addrspace(1) [[IN:%.*]], align 8
; CHECK-NEXT:    store <2 x i32> [[TMP0]], ptr addrspace(1) [[OUT:%.*]], align 8
; CHECK-NEXT:    ret void
;
entry:
  %x = load i32, ptr addrspace(1) %in, align 8
  %in1 = getelementptr i8, ptr addrspace(1) %in, i64 4
  %y = load i32, ptr addrspace(1) %in1, align 4
  store i32 %x, ptr addrspace(1) %out, align 8
  %out1 = getelementptr i8, ptr addrspace(1) %out, i64 4
  store i32 %y, ptr addrspace(1) %out1, align 4
  ret void
}

define void @pair8_transform_add_sub(ptr addrspace(1) %in, ptr addrspace(1) %out,
                                     i32 %salt) {
; CHECK-LABEL: @pair8_transform_add_sub(
; CHECK-NOT: add <2 x i32>
; CHECK-NOT: sub <2 x i32>
; CHECK: ret void
entry:
  %x = load i32, ptr addrspace(1) %in, align 8
  %in1 = getelementptr i8, ptr addrspace(1) %in, i64 4
  %y = load i32, ptr addrspace(1) %in1, align 4
  %bit = and i32 %salt, 1
  %nx = add i32 %x, %bit
  %ny = sub i32 %y, %bit
  store i32 %nx, ptr addrspace(1) %out, align 8
  %out1 = getelementptr i8, ptr addrspace(1) %out, i64 4
  store i32 %ny, ptr addrspace(1) %out1, align 4
  ret void
}
