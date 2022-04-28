; RUN: llvm-ml -m64 -filetype=s %s /Fo - | FileCheck %s --check-prefixes=CHECK-64,CHECK
; RUN: llvm-ml -m32 -filetype=s %s /Fo - | FileCheck %s --check-prefixes=CHECK-32,CHECK

.data

ifdef rax
  fn_ref qword 1
else
  fn_ref dword 1
endif

fn_ref_word word 2
fn PROC

BranchTargetStruc struc
  member0 dword ?
  ifdef rax
    member1 dword ?
  endif
BranchTargetStruc ends


ifdef rax
  fn_ref_struc BranchTargetStruc {3, 3}
else
  fn_ref_struc BranchTargetStruc {3}
endif

.code

t0:
call fn_ref
jmp fn_ref
; CHECK-LABEL: t0:
; CHECK-64: call qword ptr [rip + fn_ref]
; CHECK-64: jmp qword ptr [rip + fn_ref]
; CHECK-32: call dword ptr [fn_ref]
; CHECK-32: jmp dword ptr [fn_ref]

t1:
call [fn_ref]
jmp [fn_ref]
; CHECK-LABEL: t1:
; CHECK-64: call qword ptr [rip + fn_ref]
; CHECK-64: jmp qword ptr [rip + fn_ref]
; CHECK-32: call dword ptr [fn_ref]
; CHECK-32: jmp dword ptr [fn_ref]

ifdef rax
  t2:
  call qword ptr [fn_ref]
  jmp qword ptr [fn_ref]
  ; CHECK-64-LABEL: t2:
  ; CHECK-64: call qword ptr [rip + fn_ref]
  ; CHECK-64: jmp qword ptr [rip + fn_ref]
else
  t2:
  call dword ptr [fn_ref]
  jmp dword ptr [fn_ref]
  ; CHECK-32-LABEL: t2:
  ; CHECK-32: call dword ptr [fn_ref]
  ; CHECK-32: jmp dword ptr [fn_ref]

  t3:
  call fn_ref_word
  jmp fn_ref_word
  ; CHECK-32-LABEL: t3:
  ; CHECK-32: call word ptr [fn_ref_word]
  ; CHECK-32-NEXT: jmp word ptr [fn_ref_word]

  t4:
  call [fn_ref_word]
  jmp [fn_ref_word]
  ; CHECK-32-LABEL: t4:
  ; CHECK-32: call word ptr [fn_ref_word]
  ; CHECK-32-NEXT: jmp word ptr [fn_ref_word]

  t5:
  call word ptr [fn_ref_word]
  jmp word ptr [fn_ref_word]
  ; CHECK-32-LABEL: t5:
  ; CHECK-32: call word ptr [fn_ref_word]
  ; CHECK-32-NEXT: jmp word ptr [fn_ref_word]
endif

t6:
call t6
jmp t6
; CHECK-LABEL: t6:
; CHECK: call t6
; CHECK-NEXT: jmp t6

t7:
call [t7]
jmp [t7]
; CHECK-LABEL: t7:
; CHECK: call t7
; CHECK-NEXT: jmp t7

ifdef rax
  t8:
  call qword ptr [t8]
  jmp qword ptr [t8]
  ; CHECK-64-LABEL: t8:
  ; CHECK-64: call qword ptr [rip + t8]
  ; CHECK-64-NEXT: jmp qword ptr [rip + t8]
else
  t8:
  call dword ptr [t8]
  jmp dword ptr [t8]
  ; CHECK-32-LABEL: t8:
  ; CHECK-32: call dword ptr [t8]
  ; CHECK-32-NEXT: jmp dword ptr [t8]
endif

t9:
call fn
jmp fn
; CHECK-LABEL: t9:
; CHECK: call fn
; CHECK-NEXT: jmp fn

t10:
call [fn]
jmp [fn]
; CHECK-LABEL: t10:
; CHECK: call fn
; CHECK-NEXT: jmp fn

ifdef rax
  t11:
  call qword ptr [fn]
  jmp qword ptr [fn]
  ; CHECK-64-LABEL: t11:
  ; CHECK-64: call qword ptr [rip + fn]
  ; CHECK-64-NEXT: jmp qword ptr [rip + fn]
else
  t11:
  call dword ptr [fn]
  jmp dword ptr [fn]
  ; CHECK-32-LABEL: t11:
  ; CHECK-32: call dword ptr [fn]
  ; CHECK-32-NEXT: jmp dword ptr [fn]
endif

t12:
call fn_ref_struc
jmp fn_ref_struc
; CHECK-LABEL: t12:
; CHECK-64: call qword ptr [rip + fn_ref_struc]
; CHECK-64: jmp qword ptr [rip + fn_ref_struc]
; CHECK-32: call dword ptr [fn_ref_struc]
; CHECK-32: jmp dword ptr [fn_ref_struc]

t13:
call [fn_ref_struc]
jmp [fn_ref_struc]
; CHECK-LABEL: t13:
; CHECK-64: call qword ptr [rip + fn_ref_struc]
; CHECK-64: jmp qword ptr [rip + fn_ref_struc]
; CHECK-32: call dword ptr [fn_ref_struc]
; CHECK-32: jmp dword ptr [fn_ref_struc]

ifdef rax
  t14:
  call qword ptr [fn_ref_struc]
  jmp qword ptr [fn_ref_struc]
  ; CHECK-64-LABEL: t14:
  ; CHECK-64: call qword ptr [rip + fn_ref_struc]
  ; CHECK-64: jmp qword ptr [rip + fn_ref_struc]
else
  t14:
  call dword ptr [fn_ref_struc]
  jmp dword ptr [fn_ref_struc]
  ; CHECK-32-LABEL: t14:
  ; CHECK-32: call dword ptr [fn_ref_struc]
  ; CHECK-32: jmp dword ptr [fn_ref_struc]
endif

t15:
je t15
; CHECK-LABEL: t15:
; CHECK: je t15

t16:
je [t16];
; CHECK-LABEL: t16:
; CHECK: je t16
