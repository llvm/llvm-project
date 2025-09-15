; RUN: not llc -mtriple=amdgcn -amdgpu-enable-lower-module-lds=false < %s 2> %t.err | FileCheck %s
; RUN: FileCheck -check-prefix=ERROR %s < %t.err

; ERROR: error: unsupported expression in static initializer: addrspacecast (ptr addrspace(3) @lds.arr to ptr addrspace(4))

; CHECK: gv_flatptr_from_lds:
; CHECK-NEXT: .quad 0+32
; CHECK-NEXT: .size gv_flatptr_from_lds, 8


@lds.arr = unnamed_addr addrspace(3) global [256 x i32] poison, align 4

@gv_flatptr_from_lds = unnamed_addr addrspace(2) global ptr addrspace(4) getelementptr ([256 x i32], ptr addrspace(4) addrspacecast (ptr addrspace(3) @lds.arr to ptr addrspace(4)), i64 0, i64 8), align 4
