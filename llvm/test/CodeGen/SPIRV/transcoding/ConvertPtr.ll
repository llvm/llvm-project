; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

;; kernel void testConvertPtrToU(global int *a, global unsigned long *res) {
;;   res[0] = (unsigned long)&a[0];
;; }

; CHECK-SPIRV: OpConvertPtrToU

define dso_local spir_kernel void @testConvertPtrToU(i32 addrspace(1)* noundef %a, i64 addrspace(1)* nocapture noundef writeonly %res) local_unnamed_addr {
entry:
  %0 = ptrtoint i32 addrspace(1)* %a to i32
  %1 = zext i32 %0 to i64
  store i64 %1, i64 addrspace(1)* %res, align 8
  ret void
}

;; kernel void testConvertUToPtr(unsigned long a) {
;;   global unsigned int *res = (global unsigned int *)a;
;;   res[0] = 0;
;; }

; CHECK-SPIRV: OpConvertUToPtr

define dso_local spir_kernel void @testConvertUToPtr(i64 noundef %a) local_unnamed_addr {
entry:
  %conv = trunc i64 %a to i32
  %0 = inttoptr i32 %conv to i32 addrspace(1)*
  store i32 0, i32 addrspace(1)* %0, align 4
  ret void
}
