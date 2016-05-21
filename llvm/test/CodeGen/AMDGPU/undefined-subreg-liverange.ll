; RUN: llc -verify-machineinstrs -o /dev/null %s
; We may have subregister live ranges that are undefined on some paths. The
; verifier should not complain about this.
target triple = "amdgcn--"

define void @func() {
B0:
  br i1 undef, label %B1, label %B2

B1:
  br label %B2

B2:
  %v0 = phi <4 x float> [ zeroinitializer, %B1 ], [ <float 0.0, float 0.0, float 0.0, float undef>, %B0 ]
  br i1 undef, label %B30.1, label %B30.2

B30.1:
  %sub = fsub <4 x float> %v0, undef
  br label %B30.2

B30.2:
  %v3 = phi <4 x float> [ %sub, %B30.1 ], [ %v0, %B2 ]
  %ve0 = extractelement <4 x float> %v3, i32 0
  store float %ve0, float addrspace(3)* undef, align 4
  ret void
}
