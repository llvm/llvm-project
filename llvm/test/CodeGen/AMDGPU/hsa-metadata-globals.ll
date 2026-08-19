; The note reports the declared size of globals whose allocated size differs,
; so a host runtime resolving them through ELF symbols does not have to treat
; the redzone as part of the object.

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o /dev/null < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck %s

@padded = addrspace(1) global { i32, [28 x i8] } zeroinitializer, align 32, !sanitize.unpadded.size !0
@padded_arr = addrspace(1) global { [16 x i8], [48 x i8] } zeroinitializer, align 32, !sanitize.unpadded.size !1
@ro = addrspace(4) global { i64, [24 x i8] } zeroinitializer, align 32, !sanitize.unpadded.size !2

; No attachment means the allocated size is already the declared size.
@plain = addrspace(1) global i32 0, align 4

; A local name says nothing on its own, because another code object merged with
; this one can define a different object under it, so these are not reported.
@internal_padded = internal addrspace(1) global { [12 x i8], [52 x i8] } zeroinitializer, align 32, !sanitize.unpadded.size !3
@private_padded = private addrspace(1) global { i32, [28 x i8] } zeroinitializer, align 32, !sanitize.unpadded.size !0

; Neither a declaration nor an available_externally definition is emitted here,
; so the padding they describe belongs to the code object that defines them.
@extern_padded = external addrspace(1) global { i32, [28 x i8] }, !sanitize.unpadded.size !0
@ae_padded = available_externally addrspace(1) global { i32, [28 x i8] } zeroinitializer, align 32, !sanitize.unpadded.size !0

define amdgpu_kernel void @kern() {
  ret void
}

; The trailing amdhsa.kernels match closes the sequence, so any extra entry
; would break the chain.

; CHECK:      amdhsa.globals:
; CHECK-NEXT:   - .name: padded
; CHECK-NEXT:     .size: 4
; CHECK-NEXT:   - .name: padded_arr
; CHECK-NEXT:     .size: 16
; CHECK-NEXT:   - .name: ro
; CHECK-NEXT:     .size: 8
; CHECK-NEXT: amdhsa.kernels:

!0 = !{i64 4}
!1 = !{i64 16}
!2 = !{i64 8}
!3 = !{i64 12}
