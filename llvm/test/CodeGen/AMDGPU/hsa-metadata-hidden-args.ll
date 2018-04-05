; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX700 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX803 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX900 --check-prefix=NOTES %s

; CHECK: ---
; CHECK:  Version: [ 1, 0 ]
; CHECK:  Kernels:

; CHECK:      - Name:       test
; CHECK:        SymbolName: 'test@kd'
; CHECK:        Args:
; CHECK-NEXT:     - Name:            r
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       ValueType:       F16
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Name:            a
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       ValueType:       F16
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Name:            b
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       ValueType:       F16
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenGlobalOffsetX
; CHECK-NEXT:       ValueType:       I64
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenGlobalOffsetY
; CHECK-NEXT:       ValueType:       I64
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenGlobalOffsetZ
; CHECK-NEXT:       ValueType:       I64
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenNone
; CHECK-NEXT:       ValueType:       I8
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenNone
; CHECK-NEXT:       ValueType:       I8
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenNone
; CHECK-NEXT:       ValueType:       I8
; CHECK-NEXT:       AddrSpaceQual:   Global
define amdgpu_kernel void @test(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

!opencl.ocl.version = !{!0}
!0 = !{i32 2, i32 0}
