; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK %s

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
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Name:            a
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Name:            b
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:   CodeProps:
define amdgpu_kernel void @test0(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b) {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; CHECK:      - Name:       test8
; CHECK:        SymbolName: 'test8@kd'
; CHECK:        Args:
; CHECK-NEXT:     - Name:            r
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Name:            a
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Name:            b
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenGlobalOffsetX
; CHECK-NEXT:   CodeProps:
define amdgpu_kernel void @test8(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b) #0 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; CHECK:      - Name:       test16
; CHECK:        SymbolName: 'test16@kd'
; CHECK:        Args:
; CHECK-NEXT:     - Name:            r
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Name:            a
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Name:            b
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenGlobalOffsetX
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenGlobalOffsetY
; CHECK-NEXT:   CodeProps:
define amdgpu_kernel void @test16(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b) #1 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; CHECK:      - Name:       test24
; CHECK:        SymbolName: 'test24@kd'
; CHECK:        Args:
; CHECK-NEXT:     - Name:            r
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Name:            a
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Name:            b
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenGlobalOffsetX
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenGlobalOffsetY
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenGlobalOffsetZ
; CHECK-NEXT:   CodeProps:
define amdgpu_kernel void @test24(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b) #2 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; CHECK:      - Name:       test32
; CHECK:        SymbolName: 'test32@kd'
; CHECK:        Args:
; CHECK-NEXT:     - Name:            r
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Name:            a
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Name:            b
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenGlobalOffsetX
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenGlobalOffsetY
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenGlobalOffsetZ
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenHostcallBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:   CodeProps:
define amdgpu_kernel void @test32(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b) #3 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; CHECK:      - Name:       test48
; CHECK:        SymbolName: 'test48@kd'
; CHECK:        Args:
; CHECK-NEXT:     - Name:            r
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Name:            a
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Name:            b
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenGlobalOffsetX
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenGlobalOffsetY
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenGlobalOffsetZ
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenHostcallBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenDefaultQueue
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenCompletionAction
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:   CodeProps:
define amdgpu_kernel void @test48(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b) #4 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; CHECK:      - Name:       test56
; CHECK:        SymbolName: 'test56@kd'
; CHECK:        Args:
; CHECK-NEXT:     - Name:            r
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Name:            a
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Name:            b
; CHECK-NEXT:       Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       GlobalBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenGlobalOffsetX
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenGlobalOffsetY
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenGlobalOffsetZ
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenHostcallBuffer
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenDefaultQueue
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenCompletionAction
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:     - Size:            8
; CHECK-NEXT:       Align:           8
; CHECK-NEXT:       ValueKind:       HiddenMultiGridSyncArg
; CHECK-NEXT:       AddrSpaceQual:   Global
; CHECK-NEXT:   CodeProps:
define amdgpu_kernel void @test56(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b) #5 {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %b.val = load half, ptr addrspace(1) %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; We don't have a use of llvm.amdgcn.implicitarg.ptr, so optnone to
; avoid optimizing out the implicit argument allocation.
attributes #0 = { optnone noinline "amdgpu-implicitarg-num-bytes"="8" }
attributes #1 = { optnone noinline "amdgpu-implicitarg-num-bytes"="16" }
attributes #2 = { optnone noinline "amdgpu-implicitarg-num-bytes"="24" }
attributes #3 = { optnone noinline "amdgpu-implicitarg-num-bytes"="32" }
attributes #4 = { optnone noinline "amdgpu-implicitarg-num-bytes"="48" }
attributes #5 = { optnone noinline "amdgpu-implicitarg-num-bytes"="56" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 200}
