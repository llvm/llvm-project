; RUN: llc < %s -mtriple=armv6-linux-gnueabi | FileCheck %s -check-prefix=CHECKELF
; RUN: llc < %s -mtriple=thumbv7-apple-ios5.0 | FileCheck %s -check-prefix=CHECKT2D

%struct.A = type { i8 }
%struct.B = type { i32 }
%struct.C = type { %struct.B }
%struct.D = type { %struct.B }
%struct.E = type { %struct.B, %struct.B }

declare ptr @A_ctor_base(ptr returned)
declare ptr @B_ctor_base(ptr returned, i32)
declare ptr @B_ctor_complete(ptr returned, i32)

declare ptr @A_ctor_base_nothisret(ptr)
declare ptr @B_ctor_base_nothisret(ptr, i32)
declare ptr @B_ctor_complete_nothisret(ptr, i32)

define ptr @C_ctor_base(ptr returned %this, i32 %x) {
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
  %call = tail call ptr @A_ctor_base(ptr returned %this)
  %call2 = tail call ptr @B_ctor_base(ptr returned %this, i32 %x)
  ret ptr %this
}

define ptr @C_ctor_base_nothisret(ptr %this, i32 %x) {
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
  %call = tail call ptr @A_ctor_base_nothisret(ptr %this)
  %call2 = tail call ptr @B_ctor_base_nothisret(ptr %this, i32 %x)
  ret ptr %this
}

define ptr @C_ctor_complete(ptr %this, i32 %x) {
entry:
; CHECKELF-LABEL: C_ctor_complete:
; CHECKELF: b C_ctor_base
; CHECKT2D-LABEL: C_ctor_complete:
; CHECKT2D: b.w _C_ctor_base
  %call = tail call ptr @C_ctor_base(ptr returned %this, i32 %x)
  ret ptr %this
}

define ptr @C_ctor_complete_nothisret(ptr %this, i32 %x) {
entry:
; CHECKELF-LABEL: C_ctor_complete_nothisret:
; CHECKELF-NOT: b C_ctor_base_nothisret
; CHECKT2D-LABEL: C_ctor_complete_nothisret:
; CHECKT2D-NOT: b.w _C_ctor_base_nothisret
  %call = tail call ptr @C_ctor_base_nothisret(ptr %this, i32 %x)
  ret ptr %this
}

define ptr @D_ctor_base(ptr %this, i32 %x) {
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
  %call = tail call ptr @B_ctor_complete(ptr returned %this, i32 %x)
  %call2 = tail call ptr @B_ctor_complete(ptr returned %this, i32 %x)
  ret ptr %this
}

define ptr @E_ctor_base(ptr %this, i32 %x) {
entry:
; CHECKELF-LABEL: E_ctor_base:
; CHECKELF-NOT: b B_ctor_complete
; CHECKT2D-LABEL: E_ctor_base:
; CHECKT2D-NOT: b.w _B_ctor_complete
  %call = tail call ptr @B_ctor_complete(ptr returned %this, i32 %x)
  %b2 = getelementptr inbounds %struct.E, ptr %this, i32 0, i32 1
  %call2 = tail call ptr @B_ctor_complete(ptr returned %b2, i32 %x)
  ret ptr %this
}
