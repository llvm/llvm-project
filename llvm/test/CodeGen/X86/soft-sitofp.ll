; RUN: llc -mtriple=i386-pc-linux < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-pc-linux < %s | FileCheck %s

; Function Attrs: nounwind
; CHECK-LABEL: s64_to_d:
; CHECK: call{{l|q}} __floatdidf
define double @s64_to_d(i64 %n) #0 {
entry:
  %conv = sitofp i64 %n to double
  ret double %conv
}

; CHECK-LABEL: s64_to_f:
; CHECK: call{{l|q}} __floatdisf
define float @s64_to_f(i64 %n) #0 {
entry:
  %conv = sitofp i64 %n to float
  ret float %conv
}

; CHECK-LABEL: s32_to_d:
; CHECK: call{{l|q}} __floatsidf
define double @s32_to_d(i32 %n) #0 {
entry:
  %conv = sitofp i32 %n to double
  ret double %conv
}

; CHECK-LABEL: s32_to_f:
; CHECK: call{{l|q}} __floatsisf
define float @s32_to_f(i32 %n) #0 {
entry:
  %conv = sitofp i32 %n to float
  ret float %conv
}

; CHECK-LABEL: u64_to_d:
; CHECK: call{{l|q}} __floatundidf
define double @u64_to_d(i64 %n) #0 {
entry:
  %conv = uitofp i64 %n to double
  ret double %conv
}

; CHECK-LABEL: u64_to_f:
; CHECK: call{{l|q}} __floatundisf
define float @u64_to_f(i64 %n) #0 {
entry:
  %conv = uitofp i64 %n to float
  ret float %conv
}

; CHECK-LABEL: u32_to_d:
; CHECK: call{{l|q}} __floatunsidf
define double @u32_to_d(i32 %n) #0 {
entry:
  %conv = uitofp i32 %n to double
  ret double %conv
}

; CHECK-LABEL: u32_to_f:
; CHECK: call{{l|q}} __floatunsisf
define float @u32_to_f(i32 %n) #0 {
entry:
  %conv = uitofp i32 %n to float
  ret float %conv
}

; CHECK-LABEL: d_to_s64:
; CHECK: call{{l|q}} __fixdfdi
define i64 @d_to_s64(double %n) #0 {
entry:
  %conv = fptosi double %n to i64
  ret i64 %conv
}

; CHECK-LABEL: d_to_s32:
; CHECK: call{{l|q}} __fixdfsi
define i32 @d_to_s32(double %n) #0 {
entry:
  %conv = fptosi double %n to i32
  ret i32 %conv
}

; CHECK-LABEL: f_to_s64:
; CHECK: call{{l|q}} __fixsfdi
define i64 @f_to_s64(float %n) #0 {
entry:
  %conv = fptosi float %n to i64
  ret i64 %conv
}

; CHECK-LABEL: f_to_s32:
; CHECK: call{{l|q}} __fixsfsi
define i32 @f_to_s32(float %n) #0 {
entry:
  %conv = fptosi float %n to i32
  ret i32 %conv
}

; CHECK-LABEL: d_to_u64:
; CHECK: call{{l|q}} __fixunsdfdi
define i64 @d_to_u64(double %n) #0 {
entry:
  %conv = fptoui double %n to i64
  ret i64 %conv
}

; CHECK-LABEL: d_to_u32:
; CHECK: call{{l|q}} __fixunsdfsi
define i32 @d_to_u32(double %n) #0 {
entry:
  %conv = fptoui double %n to i32
  ret i32 %conv
}

; CHECK-LABEL: f_to_u64:
; CHECK: call{{l|q}} __fixunssfdi
define i64 @f_to_u64(float %n) #0 {
entry:
  %conv = fptoui float %n to i64
  ret i64 %conv
}

; CHECK-LABEL: f_to_u32:
; CHECK: call{{l|q}} __fixunssfsi
define i32 @f_to_u32(float %n) #0 {
entry:
  %conv = fptoui float %n to i32
  ret i32 %conv
}

attributes #0 = { nounwind "use-soft-float"="true" }
