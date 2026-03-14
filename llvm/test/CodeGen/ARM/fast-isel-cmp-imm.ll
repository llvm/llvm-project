; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-linux-gnueabi -verify-machineinstrs | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=THUMB

define void @t1a(float %a) uwtable ssp {
entry:
; ARM: t1a
; THUMB: t1a
  %cmp = fcmp oeq float %a, 0.000000e+00
; ARM: vcmp.f32 s{{[0-9]+}}, #0
; THUMB: vcmp.f32 s{{[0-9]+}}, #0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare void @foo()

; Shouldn't be able to encode -0.0 imm.
define void @t1b(float %a) uwtable ssp {
entry:
; ARM: t1b
; THUMB: t1b
  %cmp = fcmp oeq float %a, -0.000000e+00
; ARM: vldr
; ARM: vcmp.f32 s{{[0-9]+}}, s{{[0-9]+}}
; THUMB: vldr
; THUMB: vcmp.f32 s{{[0-9]+}}, s{{[0-9]+}}
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t2a(double %a) uwtable ssp {
entry:
; ARM: t2a
; THUMB: t2a
  %cmp = fcmp oeq double %a, 0.000000e+00
; ARM: vcmp.f64 d{{[0-9]+}}, #0
; THUMB: vcmp.f64 d{{[0-9]+}}, #0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Shouldn't be able to encode -0.0 imm.
define void @t2b(double %a) uwtable ssp {
entry:
; ARM: t2b
; THUMB: t2b
  %cmp = fcmp oeq double %a, -0.000000e+00
; ARM: vldr
; ARM: vcmp.f64 d{{[0-9]+}}, d{{[0-9]+}}
; THUMB: vldr
; THUMB: vcmp.f64 d{{[0-9]+}}, d{{[0-9]+}}
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t4(i8 signext %a) uwtable ssp {
entry:
; ARM: t4
; THUMB: t4
  %cmp = icmp eq i8 %a, -1
; ARM: cmn r{{[0-9]}}, #1
; THUMB: cmn.w r{{[0-9]}}, #1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t5(i8 zeroext %a) uwtable ssp {
entry:
; ARM: t5
; THUMB: t5
  %cmp = icmp eq i8 %a, 1
; ARM: cmp r{{[0-9]}}, #1
; THUMB: cmp r{{[0-9]}}, #1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t6(i16 signext %a) uwtable ssp {
entry:
; ARM: t6
; THUMB: t6
  %cmp = icmp eq i16 %a, -1
; ARM: cmn r{{[0-9]}}, #1
; THUMB: cmn.w r{{[0-9]}}, #1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t7(i16 zeroext %a) uwtable ssp {
entry:
; ARM: t7
; THUMB: t7
  %cmp = icmp eq i16 %a, 1
; ARM: cmp r{{[0-9]}}, #1
; THUMB: cmp r{{[0-9]}}, #1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t8(i32 %a) uwtable ssp {
entry:
; ARM: t8
; THUMB: t8
  %cmp = icmp eq i32 %a, -1
; ARM: cmn r{{[0-9]}}, #1
; THUMB: cmn.w r{{[0-9]}}, #1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t9(i32 %a) uwtable ssp {
entry:
; ARM: t9
; THUMB: t9
  %cmp = icmp eq i32 %a, 1
; ARM: cmp r{{[0-9]}}, #1
; THUMB: cmp r{{[0-9]}}, #1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t10(i32 %a) uwtable ssp {
entry:
; ARM: t10
; THUMB: t10
  %cmp = icmp eq i32 %a, 384
; ARM: cmp r{{[0-9]}}, #384
; THUMB: cmp.w r{{[0-9]}}, #384
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t11(i32 %a) uwtable ssp {
entry:
; ARM: t11
; THUMB: t11
  %cmp = icmp eq i32 %a, 4096
; ARM: cmp r{{[0-9]}}, #4096
; THUMB: cmp.w r{{[0-9]}}, #4096
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t12(i8 %a) uwtable ssp {
entry:
; ARM: t12
; THUMB: t12
  %cmp = icmp ugt i8 %a, -113
; ARM: cmp r{{[0-9]}}, #143
; THUMB: cmp r{{[0-9]}}, #143
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; rdar://11038907
; When comparing LONG_MIN/INT_MIN use a cmp instruction.
define void @t13() nounwind ssp {
entry:
; ARM: t13
; THUMB: t13
  %cmp = icmp slt i32 -123, -2147483648
; ARM: cmp r{{[0-9]}}, #-2147483648
; THUMB: cmp.w r{{[0-9]}}, #-2147483648
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  ret void

if.end:                                           ; preds = %entry
  ret void
}

