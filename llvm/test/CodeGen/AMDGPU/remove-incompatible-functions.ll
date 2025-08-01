; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=amdgcn -mcpu=bonaire -stop-after=amdgpu-remove-incompatible-functions\
; RUN:   -pass-remarks=amdgpu-remove-incompatible-functions < %s 2>%t | FileCheck -check-prefixes=GFX7,IR %s
; RUN: FileCheck --check-prefix=WARN-GFX7 %s < %t
; RUN: llc -mtriple=amdgcn -mcpu=bonaire < %s

; RUN: llc -enable-new-pm -mtriple=amdgcn -mcpu=bonaire -stop-after=amdgpu-remove-incompatible-functions\
; RUN:   -pass-remarks=amdgpu-remove-incompatible-functions < %s 2>%t | FileCheck -check-prefixes=GFX7,IR %s
; RUN: FileCheck --check-prefix=WARN-GFX7 %s < %t

; RUN: llc -mtriple=amdgcn -mcpu=fiji -stop-after=amdgpu-remove-incompatible-functions\
; RUN:   -pass-remarks=amdgpu-remove-incompatible-functions < %s 2>%t | FileCheck -check-prefixes=GFX8,IR %s
; RUN: FileCheck --check-prefix=WARN-GFX8 %s < %t
; RUN: llc -mtriple=amdgcn -mcpu=fiji < %s

; RUN: llc -enable-new-pm -mtriple=amdgcn -mcpu=fiji -stop-after=amdgpu-remove-incompatible-functions\
; RUN:   -pass-remarks=amdgpu-remove-incompatible-functions < %s 2>%t | FileCheck -check-prefixes=GFX8,IR %s
; RUN: FileCheck --check-prefix=WARN-GFX8 %s < %t

; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -stop-after=amdgpu-remove-incompatible-functions\
; RUN:   -pass-remarks=amdgpu-remove-incompatible-functions < %s 2>%t | FileCheck -check-prefixes=GFX9,GFX906,IR %s
; RUN: FileCheck --check-prefix=WARN-GFX906 %s < %t
; RUN: llc -mtriple=amdgcn -mcpu=gfx906 < %s

; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -stop-after=amdgpu-remove-incompatible-functions\
; RUN:   -pass-remarks=amdgpu-remove-incompatible-functions < %s 2>%t | FileCheck -check-prefixes=GFX9,GFX90A,IR %s
; RUN: FileCheck --check-prefix=WARN-GFX90A %s < %t
; RUN: llc -mtriple=amdgcn -mcpu=gfx90a < %s

; RUN: llc -mtriple=amdgcn -mcpu=gfx1011 -stop-after=amdgpu-remove-incompatible-functions\
; RUN:   -pass-remarks=amdgpu-remove-incompatible-functions < %s 2>%t | FileCheck -check-prefixes=GFX10,IR %s
; RUN: FileCheck --check-prefix=WARN-GFX10 %s < %t
; RUN: llc -mtriple=amdgcn -mcpu=gfx1011 < %s

; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -stop-after=amdgpu-remove-incompatible-functions\
; RUN:   -pass-remarks=amdgpu-remove-incompatible-functions < %s 2>%t | FileCheck -check-prefixes=GFX11,IR %s
; RUN: FileCheck --check-prefix=WARN-GFX11 %s < %t
; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 < %s

; Note: This test checks the IR, but also has a run line to codegen the file just to check we
; do not crash when trying to select those functions.

; WARN-GFX7: removing function 'needs_dpp': +dpp is not supported on the current target
; WARN-GFX7: removing function 'needs_16bit_insts': +16-bit-insts is not supported on the current target
; WARN-GFX7: removing function 'needs_gfx8_insts': +gfx8-insts is not supported on the current target
; WARN-GFX7: removing function 'needs_gfx9_insts': +gfx9-insts is not supported on the current target
; WARN-GFX7: removing function 'needs_gfx10_insts': +gfx10-insts is not supported on the current target
; WARN-GFX7: removing function 'needs_gfx11_insts': +gfx11-insts is not supported on the current target
; WARN-GFX7: removing function 'needs_dot1_insts': +dot1-insts is not supported on the current target
; WARN-GFX7: removing function 'needs_dot2_insts': +dot2-insts is not supported on the current target
; WARN-GFX7: removing function 'needs_dot3_insts': +dot3-insts is not supported on the current target
; WARN-GFX7: removing function 'needs_dot4_insts': +dot4-insts is not supported on the current target
; WARN-GFX7: removing function 'needs_dot5_insts': +dot5-insts is not supported on the current target
; WARN-GFX7: removing function 'needs_dot6_insts': +dot6-insts is not supported on the current target
; WARN-GFX7: removing function 'needs_dot7_insts': +dot7-insts is not supported on the current target
; WARN-GFX7: removing function 'needs_dot8_insts': +dot8-insts is not supported on the current target
; WARN-GFX7-NOT: not supported

; WARN-GFX8: removing function 'needs_gfx9_insts': +gfx9-insts is not supported on the current target
; WARN-GFX8: removing function 'needs_gfx10_insts': +gfx10-insts is not supported on the current target
; WARN-GFX8: removing function 'needs_gfx11_insts': +gfx11-insts is not supported on the current target
; WARN-GFX8: removing function 'needs_dot1_insts': +dot1-insts is not supported on the current target
; WARN-GFX8: removing function 'needs_dot2_insts': +dot2-insts is not supported on the current target
; WARN-GFX8: removing function 'needs_dot3_insts': +dot3-insts is not supported on the current target
; WARN-GFX8: removing function 'needs_dot4_insts': +dot4-insts is not supported on the current target
; WARN-GFX8: removing function 'needs_dot5_insts': +dot5-insts is not supported on the current target
; WARN-GFX8: removing function 'needs_dot6_insts': +dot6-insts is not supported on the current target
; WARN-GFX8: removing function 'needs_dot7_insts': +dot7-insts is not supported on the current target
; WARN-GFX8: removing function 'needs_dot8_insts': +dot8-insts is not supported on the current target
; WARN-GFX8-NOT: not supported

; WARN-GFX906: removing function 'needs_gfx10_insts': +gfx10-insts is not supported on the current target
; WARN-GFX906: removing function 'needs_gfx11_insts': +gfx11-insts is not supported on the current target
; WARN-GFX906: removing function 'needs_dot3_insts': +dot3-insts is not supported on the current target
; WARN-GFX906: removing function 'needs_dot4_insts': +dot4-insts is not supported on the current target
; WARN-GFX906: removing function 'needs_dot5_insts': +dot5-insts is not supported on the current target
; WARN-GFX906: removing function 'needs_dot6_insts': +dot6-insts is not supported on the current target
; WARN-GFX906: removing function 'needs_dot8_insts': +dot8-insts is not supported on the current target
; WARN-GFX906-NOT: not supported

; WARN-GFX90A: removing function 'needs_gfx10_insts': +gfx10-insts is not supported on the current target
; WARN-GFX90A: removing function 'needs_gfx11_insts': +gfx11-insts is not supported on the current target
; WARN-GFX90A: removing function 'needs_dot8_insts': +dot8-insts is not supported on the current target
; WARN-GFX90A-NOT: not supported

; WARN-GFX10: removing function 'needs_gfx11_insts': +gfx11-insts is not supported on the current target
; WARN-GFX10: removing function 'needs_dot3_insts': +dot3-insts is not supported on the current target
; WARN-GFX10: removing function 'needs_dot4_insts': +dot4-insts is not supported on the current target
; WARN-GFX10: removing function 'needs_dot8_insts': +dot8-insts is not supported on the current target
; WARN-GFX10-NOT: not supported

; WARN-GFX11: removing function 'needs_dot1_insts': +dot1-insts is not supported on the current target
; WARN-GFX11: removing function 'needs_dot2_insts': +dot2-insts is not supported on the current target
; WARN-GFX11: removing function 'needs_dot3_insts': +dot3-insts is not supported on the current target
; WARN-GFX11: removing function 'needs_dot4_insts': +dot4-insts is not supported on the current target
; WARN-GFX11: removing function 'needs_dot6_insts': +dot6-insts is not supported on the current target
; WARN-GFX11-NOT: not supported

; GFX7:   @GVRefs {{.*}} zeroinitializer
; GFX8:   @GVRefs {{.*}} [ptr @needs_dpp, ptr @needs_16bit_insts, ptr @needs_gfx8_insts, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null, ptr null]
; GFX906: @GVRefs {{.*}} [ptr @needs_dpp, ptr @needs_16bit_insts, ptr @needs_gfx8_insts, ptr @needs_gfx9_insts, ptr null, ptr null, ptr @needs_dot1_insts, ptr @needs_dot2_insts, ptr null, ptr null, ptr null, ptr null, ptr @needs_dot7_insts, ptr null]
; GFX90A: @GVRefs {{.*}} [ptr @needs_dpp, ptr @needs_16bit_insts, ptr @needs_gfx8_insts, ptr @needs_gfx9_insts, ptr null, ptr null, ptr @needs_dot1_insts, ptr @needs_dot2_insts, ptr @needs_dot3_insts, ptr @needs_dot4_insts, ptr @needs_dot5_insts, ptr @needs_dot6_insts, ptr @needs_dot7_insts, ptr null]
; GFX10:  @GVRefs {{.*}} [ptr @needs_dpp, ptr @needs_16bit_insts, ptr @needs_gfx8_insts, ptr @needs_gfx9_insts, ptr @needs_gfx10_insts, ptr null, ptr @needs_dot1_insts, ptr @needs_dot2_insts, ptr null, ptr null, ptr @needs_dot5_insts, ptr @needs_dot6_insts, ptr @needs_dot7_insts, ptr null]
; GFX11:  @GVRefs {{.*}} [ptr @needs_dpp, ptr @needs_16bit_insts, ptr @needs_gfx8_insts, ptr @needs_gfx9_insts, ptr @needs_gfx10_insts, ptr @needs_gfx11_insts, ptr null, ptr null, ptr null, ptr null, ptr @needs_dot5_insts, ptr null, ptr @needs_dot7_insts, ptr @needs_dot8_insts]
@GVRefs = internal global [14 x ptr] [
  ptr @needs_dpp,
  ptr @needs_16bit_insts,
  ptr @needs_gfx8_insts,
  ptr @needs_gfx9_insts,
  ptr @needs_gfx10_insts,
  ptr @needs_gfx11_insts,
  ptr @needs_dot1_insts,
  ptr @needs_dot2_insts,
  ptr @needs_dot3_insts,
  ptr @needs_dot4_insts,
  ptr @needs_dot5_insts,
  ptr @needs_dot6_insts,
  ptr @needs_dot7_insts,
  ptr @needs_dot8_insts
]

; GFX7: @ConstantExpr = internal global i64 0
@ConstantExpr = internal global i64 ptrtoint (ptr @needs_dpp to i64)

define void @needs_dpp(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c) #0 {
entry:
  %cmp = icmp eq i64 %a, 0
  br i1 %cmp, label %if, label %else

if:
  %ld = load i64, ptr %in
  br label %endif

else:
  %add = add i64 %a, %b
  br label %endif

endif:
  %phi = phi i64 [%ld, %if], [%add, %else]
  store i64 %phi, ptr %out
  ret void
}

define void @needs_16bit_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c) #1 {
entry:
  %cmp = icmp eq i64 %a, 0
  br i1 %cmp, label %if, label %else

if:
  %ld = load i64, ptr %in
  br label %endif

else:
  %add = add i64 %a, %b
  br label %endif

endif:
  %phi = phi i64 [%ld, %if], [%add, %else]
  store i64 %phi, ptr %out
  ret void
}

define void @needs_gfx8_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c) #2 {
entry:
  %cmp = icmp eq i64 %a, 0
  br i1 %cmp, label %if, label %else

if:
  %ld = load i64, ptr %in
  br label %endif

else:
  %add = add i64 %a, %b
  br label %endif

endif:
  %phi = phi i64 [%ld, %if], [%add, %else]
  store i64 %phi, ptr %out
  ret void
}

define void @needs_gfx9_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c) #3 {
entry:
  %cmp = icmp eq i64 %a, 0
  br i1 %cmp, label %if, label %else

if:
  %ld = load i64, ptr %in
  br label %endif

else:
  %add = add i64 %a, %b
  br label %endif

endif:
  %phi = phi i64 [%ld, %if], [%add, %else]
  store i64 %phi, ptr %out
  ret void
}

define void @needs_gfx10_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c) #4 {
entry:
  %cmp = icmp eq i64 %a, 0
  br i1 %cmp, label %if, label %else

if:
  %ld = load i64, ptr %in
  br label %endif

else:
  %add = add i64 %a, %b
  br label %endif

endif:
  %phi = phi i64 [%ld, %if], [%add, %else]
  store i64 %phi, ptr %out
  ret void
}

define void @needs_gfx11_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c) #5 {
entry:
  %cmp = icmp eq i64 %a, 0
  br i1 %cmp, label %if, label %else

if:
  %ld = load i64, ptr %in
  br label %endif

else:
  %add = add i64 %a, %b
  br label %endif

endif:
  %phi = phi i64 [%ld, %if], [%add, %else]
  store i64 %phi, ptr %out
  ret void
}

define void @needs_dot1_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c) #6 {
  %add = add i64 %a, %b
  store i64 %add, ptr %out
  ret void
}

define void @needs_dot2_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c) #7 {
  %add = add i64 %a, %b
  store i64 %add, ptr %out
  ret void
}

define void @needs_dot3_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c) #8 {
  %add = add i64 %a, %b
  store i64 %add, ptr %out
  ret void
}


define void @needs_dot4_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c) #9 {
  %add = add i64 %a, %b
  store i64 %add, ptr %out
  ret void
}

define void @needs_dot5_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c) #10 {
  %add = add i64 %a, %b
  store i64 %add, ptr %out
  ret void
}

define void @needs_dot6_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c) #11 {
  %add = add i64 %a, %b
  store i64 %add, ptr %out
  ret void
}

define void @needs_dot7_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c) #12 {
  %add = add i64 %a, %b
  store i64 %add, ptr %out
  ret void
}

define void @needs_dot8_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c) #13 {
  %add = add i64 %a, %b
  store i64 %add, ptr %out
  ret void
}

; IR: define void @caller(
define void @caller(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c) {
  call void @needs_dpp(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c)
  call void @needs_16bit_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c)
  call void @needs_gfx8_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c)
  ; GFX111: call void @needs_gfx9_insts(c
  call void @needs_gfx9_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c)
  ; GFX111: call void @needs_gfx10_insts(
  call void @needs_gfx10_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c)
  call void @needs_gfx11_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c)
  call void @needs_dot1_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c)
  call void @needs_dot2_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c)
  call void @needs_dot3_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c)
  call void @needs_dot4_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c)
  call void @needs_dot5_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c)
  call void @needs_dot6_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c)
  call void @needs_dot7_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c)
  call void @needs_dot8_insts(ptr %out, ptr %in, i64 %a, i64 %b, i64 %c)
  ret void
}

attributes #0 = { "target-features"="+dpp" }
attributes #1 = { "target-features"="+16-bit-insts" }
attributes #2 = { "target-features"="+gfx8-insts" }
attributes #3 = { "target-features"="+gfx9-insts" }
attributes #4 = { "target-features"="+gfx10-insts" }
attributes #5 = { "target-features"="+gfx11-insts" }
attributes #6 = { "target-features"="+dot1-insts" }
attributes #7 = { "target-features"="+dot2-insts" }
attributes #8 = { "target-features"="+dot3-insts" }
attributes #9 = { "target-features"="+dot4-insts" }
attributes #10 = { "target-features"="+dot5-insts" }
attributes #11 = { "target-features"="+dot6-insts" }
attributes #12 = { "target-features"="+dot7-insts" }
attributes #13 = { "target-features"="+dot8-insts" }
;; NOTE: These prefixes are unused and the list is autogenerated. Do not add tests below this line:
; GFX10: {{.*}}
; GFX11: {{.*}}
; GFX7: {{.*}}
; GFX8: {{.*}}
; GFX9: {{.*}}
; GFX906: {{.*}}
; GFX90A: {{.*}}
; IR: {{.*}}
