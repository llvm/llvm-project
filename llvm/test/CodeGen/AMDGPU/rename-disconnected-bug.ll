; RUN: llc -verify-machineinstrs -o /dev/null %s
; Check that renameDisconnectedComponents() does not create vregs without a
; definition on every path (there should at least be IMPLICIT_DEF instructions).
target triple = "amdgcn--"

define void @func() {
B0:
  br i1 undef, label %B1, label %B2

B1:
  br label %B2

B2:
  %v0 = phi <4 x float> [ zeroinitializer, %B1 ], [ <float 0.0, float 0.0, float 0.0, float undef>, %B0 ]
  br i1 undef, label %B20.1, label %B20.2

B20.1:
  br label %B20.2

B20.2:
  %v2 = phi <4 x float> [ zeroinitializer, %B20.1 ], [ %v0, %B2 ]
  br i1 undef, label %B30.1, label %B30.2

B30.1:
  %sub = fsub <4 x float> %v2, undef
  br label %B30.2

B30.2:
  %v3 = phi <4 x float> [ %sub, %B30.1 ], [ %v2, %B20.2 ]
  %ve0 = extractelement <4 x float> %v3, i32 0
  store float %ve0, float addrspace(3)* undef, align 4
  ret void
}
