; RUN: llc -march=mipsel -mxgot -relocation-model=pic < %s | FileCheck %s -check-prefix=O32
; RUN: llc -march=mips64el -mcpu=mips64r2 -mattr=+n64 -mxgot -relocation-model=pic < %s | \
; RUN: FileCheck %s -check-prefix=N64

@v0 = external global i32

define void @foo1() nounwind {
entry:
; O32: lui $[[R0:[0-9]+]], %got_hi(v0)
; O32: addu  $[[R1:[0-9]+]], $[[R0]], ${{[a-z0-9]+}}
; O32: lw  ${{[0-9]+}}, %got_lo(v0)($[[R1]])
; O32: lui $[[R2:[0-9]+]], %call_hi(foo0)
; O32: addu  $[[R3:[0-9]+]], $[[R2]], ${{[a-z0-9]+}}
; O32: lw  ${{[0-9]+}}, %call_lo(foo0)($[[R3]])

; N64: lui $[[R0:[0-9]+]], %got_hi(v0)
; N64: daddu  $[[R1:[0-9]+]], $[[R0]], ${{[a-z0-9]+}}
; N64: ld  ${{[0-9]+}}, %got_lo(v0)($[[R1]])
; N64: lui $[[R2:[0-9]+]], %call_hi(foo0)
; N64: daddu  $[[R3:[0-9]+]], $[[R2]], ${{[a-z0-9]+}}
; N64: ld  ${{[0-9]+}}, %call_lo(foo0)($[[R3]])

  %0 = load i32, i32* @v0, align 4
  tail call void @foo0(i32 %0) nounwind
  ret void
}

declare void @foo0(i32)

; call to external function.

define void @foo2(i32* nocapture %d, i32* nocapture %s, i32 %n) nounwind {
entry:
; O32-LABEL: foo2:
; O32: lui $[[R2:[0-9]+]], %call_hi(memcpy)
; O32: addu  $[[R3:[0-9]+]], $[[R2]], ${{[a-z0-9]+}}
; O32: lw  ${{[0-9]+}}, %call_lo(memcpy)($[[R3]])

; N64-LABEL: foo2:
; N64: lui $[[R2:[0-9]+]], %call_hi(memcpy)
; N64: daddu  $[[R3:[0-9]+]], $[[R2]], ${{[a-z0-9]+}}
; N64: ld  ${{[0-9]+}}, %call_lo(memcpy)($[[R3]])

  %0 = bitcast i32* %d to i8*
  %1 = bitcast i32* %s to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %0, i8* %1, i32 %n, i32 4, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
