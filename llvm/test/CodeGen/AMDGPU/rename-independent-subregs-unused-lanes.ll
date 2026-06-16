; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -verify-machineinstrs -filetype=null %s

; This used to fail machine verification after rename-independent-subregs
; created duplicate unused-lane subranges while repairing missing predecessor
; definitions for a split subregister component.

define <7 x i32> @multiple_predecessor_unused_lanes(<7 x i32> %ha, i32 %h.sel) {
entry:
  %h.case = and i32 %h.sel, 3
  switch i32 %h.case, label %h.default [
    i32 0, label %h.add
    i32 1, label %h.shuffle
  ]

common.ret:
  %ret = phi <7 x i32> [ %h.shuf, %h.shuffle ], [ %h.x, %h.default ]
  ret <7 x i32> %ret

h.add:
  %h.addv = or <7 x i32> %ha, splat (i32 1)
  ret <7 x i32> %h.addv

h.shuffle:
  %h.shuf = shufflevector <7 x i32> %ha, <7 x i32> %ha, <7 x i32> <i32 6, i32 0, i32 0, i32 poison, i32 0, i32 poison, i32 poison>
  br label %common.ret

h.default:
  %h.x = xor <7 x i32> %ha, <i32 0, i32 poison, i32 0, i32 0, i32 0, i32 1, i32 poison>
  br label %common.ret
}
