; RUN: llc < %s -mtriple=arm64-eabi -O0 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=arm64-eabi -O3 -verify-machineinstrs | FileCheck %s

@.str = private unnamed_addr constant [9 x i8] c"%lf %lu\0A\00", align 1
@.str1 = private unnamed_addr constant [8 x i8] c"%lf %u\0A\00", align 1
@.str2 = private unnamed_addr constant [8 x i8] c"%f %lu\0A\00", align 1
@.str3 = private unnamed_addr constant [7 x i8] c"%f %u\0A\00", align 1

define void @testDouble(double %d) ssp {
; CHECK-LABEL: testDouble:
; CHECK:  fcvtzu x{{[0-9]+}}, d{{[0-9]+}}
; CHECK:  fcvtzu w{{[0-9]+}}, d{{[0-9]+}}
entry:
  %d.addr = alloca double, align 8
  store double %d, ptr %d.addr, align 8
  %0 = load double, ptr %d.addr, align 8
  %1 = load double, ptr %d.addr, align 8
  %conv = fptoui double %1 to i64
  %call = call i32 (ptr, ...) @printf(ptr @.str, double %0, i64 %conv)
  %2 = load double, ptr %d.addr, align 8
  %3 = load double, ptr %d.addr, align 8
  %conv1 = fptoui double %3 to i32
  %call2 = call i32 (ptr, ...) @printf(ptr @.str1, double %2, i32 %conv1)
  ret void
}

declare i32 @printf(ptr, ...)

define void @testFloat(float %f) ssp {
; CHECK-LABEL: testFloat:
; CHECK:  fcvtzu x{{[0-9]+}}, s{{[0-9]+}}
; CHECK:  fcvtzu w{{[0-9]+}}, s{{[0-9]+}}
entry:
  %f.addr = alloca float, align 4
  store float %f, ptr %f.addr, align 4
  %0 = load float, ptr %f.addr, align 4
  %conv = fpext float %0 to double
  %1 = load float, ptr %f.addr, align 4
  %conv1 = fptoui float %1 to i64
  %call = call i32 (ptr, ...) @printf(ptr @.str2, double %conv, i64 %conv1)
  %2 = load float, ptr %f.addr, align 4
  %conv2 = fpext float %2 to double
  %3 = load float, ptr %f.addr, align 4
  %conv3 = fptoui float %3 to i32
  %call4 = call i32 (ptr, ...) @printf(ptr @.str3, double %conv2, i32 %conv3)
  ret void
}

define i32 @main(i32 %argc, ptr %argv) ssp {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca ptr, align 8
  store i32 0, ptr %retval
  store i32 %argc, ptr %argc.addr, align 4
  store ptr %argv, ptr %argv.addr, align 8
  call void @testDouble(double 1.159198e+01)
  call void @testFloat(float 0x40272F1800000000)
  ret i32 0
}

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"Objective-C Version", i32 2}
!1 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!2 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!3 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
