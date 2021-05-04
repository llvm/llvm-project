; RUN: llc < %s -mtriple=armv6-linux-gnueabi | FileCheck %s -check-prefix=CHECKELF
; RUN: llc < %s -mtriple=thumbv7-apple-ios5.0 | FileCheck %s -check-prefix=CHECKT2D

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
; CHECKELF-LABEL: C_ctor_base:
; CHECKELF-NOT: mov {{r[0-9]+}}, r0
; CHECKELF: bl A_ctor_base
; CHECKELF-NOT: mov r0, {{r[0-9]+}}
; CHECKELF: b B_ctor_base
; CHECKT2D-LABEL: C_ctor_base:
; CHECKT2D-NOT: mov {{r[0-9]+}}, r0
; CHECKT2D: bl _A_ctor_base
; CHECKT2D-NOT: mov r0, {{r[0-9]+}}
; CHECKT2D: b.w _B_ctor_base
  %0 = bitcast %struct.C* %this to %struct.A*
  %call = tail call %struct.A* @A_ctor_base(%struct.A* returned %0)
  %1 = getelementptr inbounds %struct.C, %struct.C* %this, i32 0, i32 0
  %call2 = tail call %struct.B* @B_ctor_base(%struct.B* returned %1, i32 %x)
  ret %struct.C* %this
}

define %struct.C* @C_ctor_base_nothisret(%struct.C* %this, i32 %x) {
entry:
; CHECKELF-LABEL: C_ctor_base_nothisret:
; CHECKELF: mov [[SAVETHIS:r[0-9]+]], r0
; CHECKELF: bl A_ctor_base_nothisret
; CHECKELF: mov r0, [[SAVETHIS]]
; CHECKELF-NOT: b B_ctor_base_nothisret
; CHECKT2D-LABEL: C_ctor_base_nothisret:
; CHECKT2D: mov [[SAVETHIS:r[0-9]+]], r0
; CHECKT2D: bl _A_ctor_base_nothisret
; CHECKT2D: mov r0, [[SAVETHIS]]
; CHECKT2D-NOT: b.w _B_ctor_base_nothisret
  %0 = bitcast %struct.C* %this to %struct.A*
  %call = tail call %struct.A* @A_ctor_base_nothisret(%struct.A* %0)
  %1 = getelementptr inbounds %struct.C, %struct.C* %this, i32 0, i32 0
  %call2 = tail call %struct.B* @B_ctor_base_nothisret(%struct.B* %1, i32 %x)
  ret %struct.C* %this
}

define %struct.C* @C_ctor_complete(%struct.C* %this, i32 %x) {
entry:
; CHECKELF-LABEL: C_ctor_complete:
; CHECKELF: b C_ctor_base
; CHECKT2D-LABEL: C_ctor_complete:
; CHECKT2D: b.w _C_ctor_base
  %call = tail call %struct.C* @C_ctor_base(%struct.C* returned %this, i32 %x)
  ret %struct.C* %this
}

define %struct.C* @C_ctor_complete_nothisret(%struct.C* %this, i32 %x) {
entry:
; CHECKELF-LABEL: C_ctor_complete_nothisret:
; CHECKELF-NOT: b C_ctor_base_nothisret
; CHECKT2D-LABEL: C_ctor_complete_nothisret:
; CHECKT2D-NOT: b.w _C_ctor_base_nothisret
  %call = tail call %struct.C* @C_ctor_base_nothisret(%struct.C* %this, i32 %x)
  ret %struct.C* %this
}

define %struct.D* @D_ctor_base(%struct.D* %this, i32 %x) {
entry:
; CHECKELF-LABEL: D_ctor_base:
; CHECKELF-NOT: mov {{r[0-9]+}}, r0
; CHECKELF: bl B_ctor_complete
; CHECKELF-NOT: mov r0, {{r[0-9]+}}
; CHECKELF: b B_ctor_complete
; CHECKT2D-LABEL: D_ctor_base:
; CHECKT2D-NOT: mov {{r[0-9]+}}, r0
; CHECKT2D: bl _B_ctor_complete
; CHECKT2D-NOT: mov r0, {{r[0-9]+}}
; CHECKT2D: b.w _B_ctor_complete
  %b = getelementptr inbounds %struct.D, %struct.D* %this, i32 0, i32 0
  %call = tail call %struct.B* @B_ctor_complete(%struct.B* returned %b, i32 %x)
  %call2 = tail call %struct.B* @B_ctor_complete(%struct.B* returned %b, i32 %x)
  ret %struct.D* %this
}

define %struct.E* @E_ctor_base(%struct.E* %this, i32 %x) {
entry:
; CHECKELF-LABEL: E_ctor_base:
; CHECKELF-NOT: b B_ctor_complete
; CHECKT2D-LABEL: E_ctor_base:
; CHECKT2D-NOT: b.w _B_ctor_complete
  %b = getelementptr inbounds %struct.E, %struct.E* %this, i32 0, i32 0
  %call = tail call %struct.B* @B_ctor_complete(%struct.B* returned %b, i32 %x)
  %b2 = getelementptr inbounds %struct.E, %struct.E* %this, i32 0, i32 1
  %call2 = tail call %struct.B* @B_ctor_complete(%struct.B* returned %b2, i32 %x)
  ret %struct.E* %this
}
