; RUN: opt -S --passes="print-dx-shader-flags" 2>&1 %s | FileCheck %s
;
; CHECK: ; Shader Flags Value: [[WAVE_FLAG:0x00080000]]
; CHECK: ; Note: shader requires additional functionality:
; CHECK-NEXT: ;       Wave level operations
; CHECK-NEXT: ; Note: extra DXIL module flags:
; CHECK-NEXT: {{^;$}}

target triple = "dxil-pc-shadermodel6.7-library"

define noundef half @wave_read_lane_at(half noundef %expr, i32 noundef %idx) {
entry:
; CHECK: Function wave_read_lane_at : [[WAVE_FLAG]]
  %ret = call half @llvm.dx.wave.readlane.f16(half %expr, i32 %idx)
  ret half %ret
}

define noundef half @wave_active_sum(half noundef %expr) {
entry:
  ; CHECK: Function wave_active_sum : [[WAVE_FLAG]]
  %ret = call half @llvm.dx.wave.reduce.sum.f16(half %expr)
  ret half %ret
}

define noundef i32 @wave_get_lane_index() {
entry:
  ; CHECK: Function wave_get_lane_index : [[WAVE_FLAG]]
  %0 = call i32 @llvm.dx.wave.getlaneindex()
  ret i32 %0
}

define noundef i1 @wave_all(i1 noundef %p1) {
entry:
  ; CHECK: Function wave_all : [[WAVE_FLAG]]
  %ret = call i1 @llvm.dx.wave.all(i1 %p1)
  ret i1 %ret
}

define noundef i1 @wave_any(i1 noundef %p1) {
entry:
  ; CHECK: Function wave_any : [[WAVE_FLAG]]
  %ret = call i1 @llvm.dx.wave.any(i1 %p1)
  ret i1 %ret
}

define noundef i32 @wave_active_countbits(i1 %expr) {
entry:
  ; CHECK: Function wave_active_countbits : [[WAVE_FLAG]]
  %0 = call i32 @llvm.dx.wave.active.countbits(i1 %expr)
  ret i32 %0
}
