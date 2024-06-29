; RUN: llc -mtriple=r600-- -mcpu=cypress < %s | FileCheck -check-prefix=EG %s

;
; kernel void combine_vloads(global char8 addrspace(5)* src, global char8 addrspace(5)* result) {
;   for (int i = 0; i < 1024; ++i)
;     result[i] = src[0] + src[1] + src[2] + src[3];
; }
;


; 128-bit loads instead of many 8-bit
; EG-LABEL: {{^}}combine_vloads:
; EG: VTX_READ_128
; EG: VTX_READ_128
define amdgpu_kernel void @combine_vloads(ptr addrspace(1) nocapture %src, ptr addrspace(1) nocapture %result) nounwind {
entry:
  br label %for.body

for.exit:                                         ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %i.01 = phi i32 [ 0, %entry ], [ %tmp19, %for.body ]
  %vecload2 = load <8 x i32>, ptr addrspace(1) %src, align 32
  %0 = bitcast <8 x i32> %vecload2 to <32 x i8>
  %tmp5 = shufflevector <32 x i8> %0, <32 x i8> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %tmp8 = shufflevector <32 x i8> %0, <32 x i8> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %tmp9 = add nsw <8 x i8> %tmp5, %tmp8
  %tmp12 = shufflevector <32 x i8> %0, <32 x i8> undef, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  %tmp13 = add nsw <8 x i8> %tmp9, %tmp12
  %tmp16 = shufflevector <32 x i8> %0, <32 x i8> undef, <8 x i32> <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %tmp17 = add nsw <8 x i8> %tmp13, %tmp16
  %scevgep = getelementptr <8 x i8>, ptr addrspace(1) %result, i32 %i.01
  %1 = bitcast <8 x i8> %tmp17 to <2 x i32>
  store <2 x i32> %1, ptr addrspace(1) %scevgep, align 8
  %tmp19 = add nsw i32 %i.01, 1
  %exitcond = icmp eq i32 %tmp19, 1024
  br i1 %exitcond, label %for.exit, label %for.body
}
