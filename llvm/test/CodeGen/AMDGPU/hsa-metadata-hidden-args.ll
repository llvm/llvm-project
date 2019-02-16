; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -mattr=-code-object-v3 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX700 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -mattr=-code-object-v3 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX803 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-code-object-v3 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX900 --check-prefix=NOTES %s

; CHECK: ---
; CHECK:  Version: [ 1, 0 ]
; CHECK:  Kernels:

; CHECK:      - Name:       test0
; CHECK:        SymbolName: 'test0@kd'
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
; CHECK-NEXT:   CodeProps:
define amdgpu_kernel void @test0(
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

; CHECK:      - Name:       test8
; CHECK:        SymbolName: 'test8@kd'
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
; CHECK-NEXT:   CodeProps:
define amdgpu_kernel void @test8(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) #0 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; CHECK:      - Name:       test16
; CHECK:        SymbolName: 'test16@kd'
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
; CHECK-NEXT:   CodeProps:
define amdgpu_kernel void @test16(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) #1 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; CHECK:      - Name:       test24
; CHECK:        SymbolName: 'test24@kd'
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
; CHECK-NEXT:   CodeProps:
define amdgpu_kernel void @test24(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) #2 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; CHECK:      - Name:       test32
; CHECK:        SymbolName: 'test32@kd'
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
; CHECK-NEXT:   CodeProps:
define amdgpu_kernel void @test32(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) #3 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

; CHECK:      - Name:       test48
; CHECK:        SymbolName: 'test48@kd'
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
; CHECK-NEXT:   CodeProps:
define amdgpu_kernel void @test48(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) #4 {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, half addrspace(1)* %r
  ret void
}

attributes #0 = { "amdgpu-implicitarg-num-bytes"="8" }
attributes #1 = { "amdgpu-implicitarg-num-bytes"="16" }
attributes #2 = { "amdgpu-implicitarg-num-bytes"="24" }
attributes #3 = { "amdgpu-implicitarg-num-bytes"="32" }
attributes #4 = { "amdgpu-implicitarg-num-bytes"="48" }
