; RUN: llc < %s -march=arm64 -aarch64-this-return-forwarding | FileCheck %s

%struct.A = type { i8 }
%struct.B = type { i32 }
%struct.C = type { %struct.B }
%struct.D = type { %struct.B }
%struct.E = type { %struct.B, %struct.B }

declare %struct.A* @A_ctor_base(%struct.A* returned)
declare %struct.B* @B_ctor_base(%struct.B* returned, i32)
declare %struct.B* @B_ctor_complete(%struct.B* returned, i32)

declare %struct.A* @A_ctor_base_nothisret(%struct.A*)
declare %struct.B* @B_ctor_base_nothisret(%struct.B*, i32)
declare %struct.B* @B_ctor_complete_nothisret(%struct.B*, i32)

define %struct.C* @C_ctor_base(%struct.C* returned %this, i32 %x) {
entry:
; CHECK-LABEL: C_ctor_base:
; CHECK-NOT: mov {{x[0-9]+}}, x0
; CHECK: bl {{_?A_ctor_base}}
; CHECK-NOT: mov x0, {{x[0-9]+}}
; CHECK: b {{_?B_ctor_base}}
  %0 = bitcast %struct.C* %this to %struct.A*
  %call = tail call %struct.A* @A_ctor_base(%struct.A* %0)
  %1 = getelementptr inbounds %struct.C, %struct.C* %this, i32 0, i32 0
  %call2 = tail call %struct.B* @B_ctor_base(%struct.B* %1, i32 %x)
  ret %struct.C* %this
}

define %struct.C* @C_ctor_base_nothisret(%struct.C* %this, i32 %x) {
entry:
; CHECK-LABEL: C_ctor_base_nothisret:
; CHECK: mov [[SAVETHIS:x[0-9]+]], x0
; CHECK: bl {{_?A_ctor_base_nothisret}}
; CHECK: mov x0, [[SAVETHIS]]
; CHECK-NOT: b {{_?B_ctor_base_nothisret}}
  %0 = bitcast %struct.C* %this to %struct.A*
  %call = tail call %struct.A* @A_ctor_base_nothisret(%struct.A* %0)
  %1 = getelementptr inbounds %struct.C, %struct.C* %this, i32 0, i32 0
  %call2 = tail call %struct.B* @B_ctor_base_nothisret(%struct.B* %1, i32 %x)
  ret %struct.C* %this
}

define %struct.C* @C_ctor_complete(%struct.C* %this, i32 %x) {
entry:
; CHECK-LABEL: C_ctor_complete:
; CHECK: b {{_?C_ctor_base}}
  %call = tail call %struct.C* @C_ctor_base(%struct.C* %this, i32 %x)
  ret %struct.C* %this
}

define %struct.C* @C_ctor_complete_nothisret(%struct.C* %this, i32 %x) {
entry:
; CHECK-LABEL: C_ctor_complete_nothisret:
; CHECK-NOT: b {{_?C_ctor_base_nothisret}}
  %call = tail call %struct.C* @C_ctor_base_nothisret(%struct.C* %this, i32 %x)
  ret %struct.C* %this
}

define %struct.D* @D_ctor_base(%struct.D* %this, i32 %x) {
entry:
; CHECK-LABEL: D_ctor_base:
; CHECK-NOT: mov {{x[0-9]+}}, x0
; CHECK: bl {{_?B_ctor_complete}}
; CHECK-NOT: mov x0, {{x[0-9]+}}
; CHECK: b {{_?B_ctor_complete}}
  %b = getelementptr inbounds %struct.D, %struct.D* %this, i32 0, i32 0
  %call = tail call %struct.B* @B_ctor_complete(%struct.B* %b, i32 %x)
  %call2 = tail call %struct.B* @B_ctor_complete(%struct.B* %b, i32 %x)
  ret %struct.D* %this
}

define %struct.E* @E_ctor_base(%struct.E* %this, i32 %x) {
entry:
; CHECK-LABEL: E_ctor_base:
; CHECK-NOT: b {{_?B_ctor_complete}}
  %b = getelementptr inbounds %struct.E, %struct.E* %this, i32 0, i32 0
  %call = tail call %struct.B* @B_ctor_complete(%struct.B* %b, i32 %x)
  %b2 = getelementptr inbounds %struct.E, %struct.E* %this, i32 0, i32 1
  %call2 = tail call %struct.B* @B_ctor_complete(%struct.B* %b2, i32 %x)
  ret %struct.E* %this
}
