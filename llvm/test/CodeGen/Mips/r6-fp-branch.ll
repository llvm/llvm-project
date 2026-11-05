; RUN: llc -mtriple=mips -mcpu=mips32r6 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not='{{^[[:blank:]]+(j|bc)[[:blank:]]}}' --implicit-check-not=mfc1 --implicit-check-not=andi --implicit-check-not="not $"
; RUN: llc -mtriple=mipsel -mcpu=mips32r6 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not='{{^[[:blank:]]+(j|bc)[[:blank:]]}}' --implicit-check-not=mfc1 --implicit-check-not=andi --implicit-check-not="not $"
; RUN: llc -mtriple=mips -mcpu=mips32r6 -mattr=+micromips -verify-machineinstrs < %s | FileCheck %s --implicit-check-not='{{^[[:blank:]]+(j|bc)[[:blank:]]}}' --implicit-check-not=mfc1 --implicit-check-not=andi --implicit-check-not="not $"
; RUN: llc -mtriple=mipsel -mcpu=mips32r6 -mattr=+micromips -verify-machineinstrs < %s | FileCheck %s --implicit-check-not='{{^[[:blank:]]+(j|bc)[[:blank:]]}}' --implicit-check-not=mfc1 --implicit-check-not=andi --implicit-check-not="not $"
; RUN: llc -mtriple=mips64 -mcpu=mips64r6 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not='{{^[[:blank:]]+(j|bc)[[:blank:]]}}' --implicit-check-not=mfc1 --implicit-check-not=andi --implicit-check-not="not $"
; RUN: llc -mtriple=mips64el -mcpu=mips64r6 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not='{{^[[:blank:]]+(j|bc)[[:blank:]]}}' --implicit-check-not=mfc1 --implicit-check-not=andi --implicit-check-not="not $"
; RUN: llc -mtriple=mips64 -mcpu=mips64r6 -target-abi=n32 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not='{{^[[:blank:]]+(j|bc)[[:blank:]]}}' --implicit-check-not=mfc1 --implicit-check-not=andi --implicit-check-not="not $"
; RUN: llc -mtriple=mips64el -mcpu=mips64r6 -target-abi=n32 -verify-machineinstrs < %s | FileCheck %s --implicit-check-not='{{^[[:blank:]]+(j|bc)[[:blank:]]}}' --implicit-check-not=mfc1 --implicit-check-not=andi --implicit-check-not="not $"

; Branches test bit zero of the FP comparison mask, without an integer copy
; or normalization to 0/1. Branch analysis must also eliminate redundant jumps.
; Cover every nonconstant predicate, including unordered and inverted forms.
declare void @true_target()
declare void @false_target()

define void @f32_oeq(float %a, float %b) {
; CHECK-LABEL: f32_oeq:
; CHECK: cmp.eq.s $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1eqz{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp oeq float %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f32_ogt(float %a, float %b) {
; CHECK-LABEL: f32_ogt:
; CHECK: cmp.ule.s $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp ogt float %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f32_oge(float %a, float %b) {
; CHECK-LABEL: f32_oge:
; CHECK: cmp.ult.s $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp oge float %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f32_olt(float %a, float %b) {
; CHECK-LABEL: f32_olt:
; CHECK: cmp.ule.s $f[[CC:[0-9]+]], $f{{13|14}}, $f12
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp olt float %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f32_ole(float %a, float %b) {
; CHECK-LABEL: f32_ole:
; CHECK: cmp.ult.s $f[[CC:[0-9]+]], $f{{13|14}}, $f12
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp ole float %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f32_one(float %a, float %b) {
; CHECK-LABEL: f32_one:
; CHECK: cmp.ueq.s $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp one float %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f32_ord(float %a, float %b) {
; CHECK-LABEL: f32_ord:
; CHECK: cmp.un.s $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp ord float %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f32_ueq(float %a, float %b) {
; CHECK-LABEL: f32_ueq:
; CHECK: cmp.ueq.s $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %t
  %c = fcmp ueq float %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f32_ugt(float %a, float %b) {
; CHECK-LABEL: f32_ugt:
; CHECK: cmp.le.s $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp ugt float %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f32_uge(float %a, float %b) {
; CHECK-LABEL: f32_uge:
; CHECK: cmp.lt.s $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp uge float %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f32_ult(float %a, float %b) {
; CHECK-LABEL: f32_ult:
; CHECK: cmp.le.s $f[[CC:[0-9]+]], $f{{13|14}}, $f12
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp ult float %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f32_ule(float %a, float %b) {
; CHECK-LABEL: f32_ule:
; CHECK: cmp.lt.s $f[[CC:[0-9]+]], $f{{13|14}}, $f12
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp ule float %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f32_une(float %a, float %b) {
; CHECK-LABEL: f32_une:
; CHECK: cmp.eq.s $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp une float %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f32_uno(float %a, float %b) {
; CHECK-LABEL: f32_uno:
; CHECK: cmp.un.s $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %t
  %c = fcmp uno float %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f64_oeq(double %a, double %b) {
; CHECK-LABEL: f64_oeq:
; CHECK: cmp.eq.d $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1eqz{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp oeq double %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f64_ogt(double %a, double %b) {
; CHECK-LABEL: f64_ogt:
; CHECK: cmp.ule.d $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp ogt double %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f64_oge(double %a, double %b) {
; CHECK-LABEL: f64_oge:
; CHECK: cmp.ult.d $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp oge double %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f64_olt(double %a, double %b) {
; CHECK-LABEL: f64_olt:
; CHECK: cmp.ule.d $f[[CC:[0-9]+]], $f{{13|14}}, $f12
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp olt double %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f64_ole(double %a, double %b) {
; CHECK-LABEL: f64_ole:
; CHECK: cmp.ult.d $f[[CC:[0-9]+]], $f{{13|14}}, $f12
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp ole double %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f64_one(double %a, double %b) {
; CHECK-LABEL: f64_one:
; CHECK: cmp.ueq.d $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp one double %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f64_ord(double %a, double %b) {
; CHECK-LABEL: f64_ord:
; CHECK: cmp.un.d $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp ord double %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f64_ueq(double %a, double %b) {
; CHECK-LABEL: f64_ueq:
; CHECK: cmp.ueq.d $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %t
  %c = fcmp ueq double %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f64_ugt(double %a, double %b) {
; CHECK-LABEL: f64_ugt:
; CHECK: cmp.le.d $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp ugt double %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f64_uge(double %a, double %b) {
; CHECK-LABEL: f64_uge:
; CHECK: cmp.lt.d $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp uge double %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f64_ult(double %a, double %b) {
; CHECK-LABEL: f64_ult:
; CHECK: cmp.le.d $f[[CC:[0-9]+]], $f{{13|14}}, $f12
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp ult double %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f64_ule(double %a, double %b) {
; CHECK-LABEL: f64_ule:
; CHECK: cmp.lt.d $f[[CC:[0-9]+]], $f{{13|14}}, $f12
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp ule double %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f64_une(double %a, double %b) {
; CHECK-LABEL: f64_une:
; CHECK: cmp.eq.d $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %f
  %c = fcmp une double %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}

define void @f64_uno(double %a, double %b) {
; CHECK-LABEL: f64_uno:
; CHECK: cmp.un.d $f[[CC:[0-9]+]], $f12, $f{{13|14}}
; CHECK: bc1nez{{c?}} $f[[CC]], [[DEST:[.$A-Za-z0-9_]+]]
; CHECK: [[DEST]]: # %t
  %c = fcmp uno double %a, %b
  br i1 %c, label %t, label %f
t:
  call void @true_target()
  ret void
f:
  call void @false_target()
  ret void
}
