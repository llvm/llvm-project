; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

; Make sure reduceBuildVecExtToExtBuildVec combine doesn't regress

; code with legal v4i16. The v4i16 build_vector it produces will be
; custom lowered into an i32 based build_vector, producing a mess that
; nothing manages to put back together.

; GCN-LABEL: {{^}}v2i16_to_i64:
; GFX9: s_waitcnt
; GFX9-NEXT: v_pk_add_u16 v1, v0, v1
; GFX9-NEXT: v_and_b32_e32 v0, 0xffff, v1
; GFX9-NEXT: v_lshrrev_b32_e32 v1, 16, v1
; GFX9-NEXT: s_setpc_b64
define i64 @v2i16_to_i64(<2 x i16> %x, <2 x i16> %y) {
  %x.add = add <2 x i16> %x, %y
  %zext = zext <2 x i16> %x.add to <2 x i32>
  %arst = bitcast <2 x i32> %zext to i64
  ret i64 %arst
}
