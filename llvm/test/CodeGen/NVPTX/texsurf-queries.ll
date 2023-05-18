; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs | FileCheck %s --check-prefix=SM20
; RUN: llc < %s -march=nvptx64 -mcpu=sm_30 -verify-machineinstrs | FileCheck %s --check-prefix=SM30
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_30 -verify-machineinstrs | %ptxas-verify %}

target triple = "nvptx-unknown-cuda"

@tex0 = internal addrspace(1) global i64 0, align 8
@surf0 = internal addrspace(1) global i64 0, align 8

declare i32 @llvm.nvvm.txq.width(i64)
declare i32 @llvm.nvvm.txq.height(i64)
declare i32 @llvm.nvvm.suq.width(i64)
declare i32 @llvm.nvvm.suq.height(i64)
declare i64 @llvm.nvvm.texsurf.handle.internal.p1(ptr addrspace(1))


; SM20-LABEL: @t0
; SM30-LABEL: @t0
define i32 @t0(i64 %texHandle) {
; SM20: txq.width.b32
; SM30: txq.width.b32
  %width = tail call i32 @llvm.nvvm.txq.width(i64 %texHandle)
  ret i32 %width
}

; SM20-LABEL: @t1
; SM30-LABEL: @t1
define i32 @t1() {
; SM30: mov.u64 %rd[[HANDLE:[0-9]+]], tex0
  %texHandle = tail call i64 @llvm.nvvm.texsurf.handle.internal.p1(ptr addrspace(1) @tex0)
; SM20: txq.width.b32 %r{{[0-9]+}}, [tex0]
; SM30: txq.width.b32 %r{{[0-9]+}}, [%rd[[HANDLE:[0-9]+]]]
  %width = tail call i32 @llvm.nvvm.txq.width(i64 %texHandle)
  ret i32 %width
}


; SM20-LABEL: @t2
; SM30-LABEL: @t2
define i32 @t2(i64 %texHandle) {
; SM20: txq.height.b32
; SM30: txq.height.b32
  %height = tail call i32 @llvm.nvvm.txq.height(i64 %texHandle)
  ret i32 %height
}

; SM20-LABEL: @t3
; SM30-LABEL: @t3
define i32 @t3() {
; SM30: mov.u64 %rd[[HANDLE:[0-9]+]], tex0
  %texHandle = tail call i64 @llvm.nvvm.texsurf.handle.internal.p1(ptr addrspace(1) @tex0)
; SM20: txq.height.b32 %r{{[0-9]+}}, [tex0]
; SM30: txq.height.b32 %r{{[0-9]+}}, [%rd[[HANDLE:[0-9]+]]]
  %height = tail call i32 @llvm.nvvm.txq.height(i64 %texHandle)
  ret i32 %height
}


; SM20-LABEL: @s0
; SM30-LABEL: @s0
define i32 @s0(i64 %surfHandle) {
; SM20: suq.width.b32
; SM30: suq.width.b32
  %width = tail call i32 @llvm.nvvm.suq.width(i64 %surfHandle)
  ret i32 %width
}

; SM20-LABEL: @s1
; SM30-LABEL: @s1
define i32 @s1() {
; SM30: mov.u64 %rd[[HANDLE:[0-9]+]], surf0
  %surfHandle = tail call i64 @llvm.nvvm.texsurf.handle.internal.p1(ptr addrspace(1) @surf0)
; SM20: suq.width.b32 %r{{[0-9]+}}, [surf0]
; SM30: suq.width.b32 %r{{[0-9]+}}, [%rd[[HANDLE:[0-9]+]]]
  %width = tail call i32 @llvm.nvvm.suq.width(i64 %surfHandle)
  ret i32 %width
}


; SM20-LABEL: @s2
; SM30-LABEL: @s2
define i32 @s2(i64 %surfHandle) {
; SM20: suq.height.b32
; SM30: suq.height.b32
  %height = tail call i32 @llvm.nvvm.suq.height(i64 %surfHandle)
  ret i32 %height
}

; SM20-LABEL: @s3
; SM30-LABEL: @s3
define i32 @s3() {
; SM30: mov.u64 %rd[[HANDLE:[0-9]+]], surf0
  %surfHandle = tail call i64 @llvm.nvvm.texsurf.handle.internal.p1(ptr addrspace(1) @surf0)
; SM20: suq.height.b32 %r{{[0-9]+}}, [surf0]
; SM30: suq.height.b32 %r{{[0-9]+}}, [%rd[[HANDLE:[0-9]+]]]
  %height = tail call i32 @llvm.nvvm.suq.height(i64 %surfHandle)
  ret i32 %height
}



!nvvm.annotations = !{!1, !2}
!1 = !{ptr addrspace(1) @tex0, !"texture", i32 1}
!2 = !{ptr addrspace(1) @surf0, !"surface", i32 1}
