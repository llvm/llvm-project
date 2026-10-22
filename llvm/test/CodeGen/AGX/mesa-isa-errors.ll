; Instructions Mesa's compiler does not emit are errors in strict mode.
; RUN: not llc -mtriple=agx -mcpu=agx-g14c -mattr=+agx-strict < %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=agx -mcpu=agx-g14c -mattr=+agx-strict -filetype=obj < %s -o /dev/null 2>&1 | FileCheck %s
; Without the feature the same code compiles.
; RUN: llc -mtriple=agx -mcpu=agx-g14c < %s -o /dev/null
; Mesa compiles only for G13 and G14.
; RUN: not --crash llc -mtriple=agx -mcpu=agx-g16g -mattr=+agx-strict < %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=APPLE9

; Special register 17 is not one Mesa reads.
; CHECK: error: GET_SR_g13 is not an instruction Mesa emits (agx-strict): get_sr: field value
; APPLE9: LLVM ERROR: the agx-strict feature needs a G13 or G14 processor, not agx-g16g
declare i32 @llvm.agx.read.sr(i32)
define void @sr17(ptr addrspace(1) %p) {
  %v = call i32 @llvm.agx.read.sr(i32 17)
  store i32 %v, ptr addrspace(1) %p
  ret void
}
