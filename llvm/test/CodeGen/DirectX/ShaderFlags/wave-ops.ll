; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
;
; Test that we have the correct shader flags to indicate that there are wave
; ops set at the module level
;
; CHECK: ; Shader Flags Value: [[WAVE_FLAG:0x00080000]]
; CHECK: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       Wave level operations
; CHECK-NEXT: ; Note: extra DXIL module flags:

target triple = "dxil-pc-shadermodel6.7-library"
%dx.types.fouri32 = type { i32, i32, i32, i32 }

; Test the indiviual ops that they have the same Shader Wave flag at the
; function level to ensure that each op is setting it accordingly

define noundef i1 @wave_is_first_lane() {
entry:
  ; CHECK: Function wave_is_first_lane : [[WAVE_FLAG]]
  %ret = call i1 @llvm.dx.wave.is.first.lane()
  ret i1 %ret
}

define noundef i32 @wave_getlaneindex() {
entry:
  ; CHECK: Function wave_getlaneindex : [[WAVE_FLAG]]
  %ret = call i32 @llvm.dx.wave.getlaneindex()
  ret i32 %ret
}

define noundef i1 @wave_any(i1 %x) {
entry:
  ; CHECK: Function wave_any : [[WAVE_FLAG]]
  %ret = call i1 @llvm.dx.wave.any(i1 %x)
  ret i1 %ret
}

define noundef i1 @wave_all(i1 %x) {
entry:
  ; CHECK: Function wave_all : [[WAVE_FLAG]]
  %ret = call i1 @llvm.dx.wave.all(i1 %x)
  ret i1 %ret
}

define noundef i1 @wave_readlane(i1 %x, i32 %idx) {
entry:
  ; CHECK: Function wave_readlane : [[WAVE_FLAG]]
  %ret = call i1 @llvm.dx.wave.readlane.i1(i1 %x, i32 %idx)
  ret i1 %ret
}

define noundef i32 @wave_reduce_sum(i32 noundef %x) {
entry:
  ; CHECK: Function wave_reduce_sum : [[WAVE_FLAG]]
  %ret = call i32 @llvm.dx.wave.reduce.sum.i32(i32 %x)
  ret i32 %ret
}

define noundef i32 @wave_reduce_usum(i32 noundef %x) {
entry:
  ; CHECK: Function wave_reduce_usum : [[WAVE_FLAG]]
  %ret = call i32 @llvm.dx.wave.reduce.usum.i32(i32 %x)
  ret i32 %ret
}

define noundef i32 @wave_reduce_max(i32 noundef %x) {
entry:
  ; CHECK: Function wave_reduce_max : [[WAVE_FLAG]]
  %ret = call i32 @llvm.dx.wave.reduce.max.i32(i32 %x)
  ret i32 %ret
}

define noundef i32 @wave_reduce_umax(i32 noundef %x) {
entry:
  ; CHECK: Function wave_reduce_umax : [[WAVE_FLAG]]
  %ret = call i32 @llvm.dx.wave.reduce.umax.i32(i32 %x)
  ret i32 %ret
}

define noundef i32 @wave_reduce_min(i32 noundef %x) {
entry:
  ; CHECK: Function wave_reduce_min : [[WAVE_FLAG]]
  %ret = call i32 @llvm.dx.wave.reduce.min.i32(i32 %x)
  ret i32 %ret
}

define noundef i32 @wave_reduce_umin(i32 noundef %x) {
entry:
  ; CHECK: Function wave_reduce_umin : [[WAVE_FLAG]]
  %ret = call i32 @llvm.dx.wave.reduce.umin.i32(i32 %x)
  ret i32 %ret
}

define void @wave_active_countbits(i1 %expr) {
entry:
  ; CHECK: Function wave_active_countbits : [[WAVE_FLAG]]
  %0 = call i32 @llvm.dx.wave.active.countbits(i1 %expr)
  ret void
}

define void @wave_active_ballot(i1 %expr) {
entry:
  ; CHECK: Function wave_active_ballot : [[WAVE_FLAG]]
  %0 = call %dx.types.fouri32 @llvm.dx.wave.ballot(i1 %expr)
  ret void
}
