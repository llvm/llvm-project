; FIXME: FastISel currently returns false if it hits code that uses VSX
; registers and with -fast-isel-abort=1 turned on the test case will then fail.
; When fastisel better supports VSX fix up this test case.
;
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=-vsx | FileCheck %s --check-prefix=ELF64
define void @t1a(float %a) nounwind {
entry:
; ELF64: t1a
  %cmp = fcmp oeq float %a, 0.000000e+00
; ELF64: addis
; ELF64: lfs
; ELF64: fcmpu
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare void @foo()

define void @t1b(float %a) nounwind {
entry:
; ELF64: t1b
  %cmp = fcmp oeq float %a, -0.000000e+00
; ELF64: addis
; ELF64: lfs
; ELF64: fcmpu
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t2a(double %a) nounwind {
entry:
; ELF64: t2a
  %cmp = fcmp oeq double %a, 0.000000e+00
; ELF64: addis
; ELF64: lfd
; ELF64: fcmpu
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t2b(double %a) nounwind {
entry:
; ELF64: t2b
  %cmp = fcmp oeq double %a, -0.000000e+00
; ELF64: addis
; ELF64: lfd
; ELF64: fcmpu
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t4(i8 signext %a) nounwind {
entry:
; ELF64: t4
  %cmp = icmp eq i8 %a, -1
; ELF64: extsb
; ELF64: cmpwi
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t5(i8 zeroext %a) nounwind {
entry:
; ELF64: t5
  %cmp = icmp eq i8 %a, 1
; ELF64: extsb
; ELF64: cmpwi
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t6(i16 signext %a) nounwind {
entry:
; ELF64: t6
  %cmp = icmp eq i16 %a, -1
; ELF64: extsh
; ELF64: cmpwi
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t7(i16 zeroext %a) nounwind {
entry:
; ELF64: t7
  %cmp = icmp eq i16 %a, 1
; ELF64: extsh
; ELF64: cmpwi
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t8(i32 %a) nounwind {
entry:
; ELF64: t8
  %cmp = icmp eq i32 %a, -1
; ELF64: cmpwi
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t9(i32 %a) nounwind {
entry:
; ELF64: t9
  %cmp = icmp eq i32 %a, 1
; ELF64: cmpwi
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t10(i32 %a) nounwind {
entry:
; ELF64: t10
  %cmp = icmp eq i32 %a, 384
; ELF64: cmpwi
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t11(i32 %a) nounwind {
entry:
; ELF64: t11
  %cmp = icmp eq i32 %a, 4096
; ELF64: cmpwi
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t12(i8 %a) nounwind {
entry:
; ELF64: t12
  %cmp = icmp ugt i8 %a, -113
; ELF64: clrlwi
; ELF64: cmplwi
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t13() nounwind ssp {
entry:
; ELF64: t13
  %cmp = icmp slt i32 -123, -2147483648
; ELF64: li
; ELF64: lis
; ELF64: cmpw
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  ret void

if.end:                                           ; preds = %entry
  ret void
}

define void @t14(i64 %a) nounwind {
entry:
; ELF64: t14
  %cmp = icmp eq i64 %a, -1
; ELF64: cmpdi
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t15(i64 %a) nounwind {
entry:
; ELF64: t15
  %cmp = icmp eq i64 %a, 1
; ELF64: cmpdi
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t16(i64 %a) nounwind {
entry:
; ELF64: t16
  %cmp = icmp eq i64 %a, 384
; ELF64: cmpdi
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t17(i64 %a) nounwind {
entry:
; ELF64: t17
  %cmp = icmp eq i64 %a, 32768
; Extra operand so we don't match on cmpdi.
; ELF64: cmpd {{[0-9]+}}
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

