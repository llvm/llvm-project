; RUN: llc -march=mipsel -mattr=+xgot \
; RUN:     -relocation-model=pic < %s | FileCheck %s -check-prefix=O32
; RUN: llc -march=mips64el -mcpu=mips64r2 -mattr=+xgot \
; RUN:     -relocation-model=pic < %s | FileCheck %s -check-prefix=N64
; RUN: llc -march=mipsel -mattr=+xgot -fast-isel \
; RUN:     -relocation-model=pic < %s | FileCheck %s -check-prefix=O32
; RUN: llc -march=mips64el -mcpu=mips64r2 -mattr=+xgot -fast-isel \
; RUN:     -relocation-model=pic < %s | FileCheck %s -check-prefix=N64

@v0 = external global i32

define void @foo1() nounwind {
entry:
; O32-LABEL: foo1:
; O32: lui $[[R0:[0-9]+]], %got_hi(v0)
; O32: addu  $[[R1:[0-9]+]], $[[R0]], ${{[a-z0-9]+}}
; O32: lw  ${{[0-9]+}}, %got_lo(v0)($[[R1]])
; O32: lui $[[R2:[0-9]+]], %call_hi(foo0)
; O32: addu  $[[R3:[0-9]+]], $[[R2]], ${{[a-z0-9]+}}
; O32: lw  ${{[0-9]+}}, %call_lo(foo0)($[[R3]])

; N64-LABEL: foo1:
; N64-DAG: lui $[[R0:[0-9]+]], %got_hi(v0)
; N64-DAG: daddu  $[[R1:[0-9]+]], $[[R0]], ${{[a-z0-9]+}}
; N64-DAG: lui $[[R2:[0-9]+]], %call_hi(foo0)
; N64-DAG: daddu  $[[R3:[0-9]+]], $[[R2]], ${{[a-z0-9]+}}
; N64-DAG: ld  ${{[0-9]+}}, %got_lo(v0)($[[R1]])
; N64-DAG: ld  ${{[0-9]+}}, %call_lo(foo0)($[[R3]])

  %0 = load i32, ptr @v0, align 4
  tail call void @foo0(i32 %0) nounwind
  ret void
}

declare void @foo0(i32)

; call to external function.

define void @foo2(ptr nocapture %d, ptr nocapture %s, i32 %n) nounwind {
entry:
; O32-LABEL: foo2:
; O32: lui $[[R2:[0-9]+]], %call_hi(memcpy)
; O32: addu  $[[R3:[0-9]+]], $[[R2]], ${{[a-z0-9]+}}
; O32: lw  ${{[0-9]+}}, %call_lo(memcpy)($[[R3]])

; N64-LABEL: foo2:
; N64: lui $[[R2:[0-9]+]], %call_hi(memcpy)
; N64: daddu  $[[R3:[0-9]+]], $[[R2]], ${{[a-z0-9]+}}
; N64: ld  ${{[0-9]+}}, %call_lo(memcpy)($[[R3]])

  tail call void @llvm.memcpy.p0.p0.i32(ptr align 4 %d, ptr align 4 %s, i32 %n, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
