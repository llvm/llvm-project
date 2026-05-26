; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=-vsx | FileCheck %s --check-prefix=ELF64
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64le-unknown-linux-gnu -mattr=+vsx | FileCheck %s --check-prefix=VSX
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc-unknown-linux-gnu -mcpu=e500 -mattr=spe | FileCheck %s --check-prefix=SPE

declare void @foo()

define void @t1a(float %a) nounwind {
entry:
; ELF64-LABEL: @t1a
; SPE-LABEL: @t1a
; VSX-LABEL: @t1a
  %cmp = fcmp oeq float %a, 0.000000e+00
; ELF64: addis
; ELF64: lfs
; ELF64: fcmpu
; VSX: addis
; VSX: lfs
; VSX: fcmpu
; SPE: efscmpeq
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t1b(float %a) nounwind {
entry:
; ELF64-LABEL: @t1b
; SPE-LABEL: @t1b
; VSX-LABEL: @t1b
  %cmp = fcmp oeq float %a, -0.000000e+00
; ELF64: addis
; ELF64: lfs
; ELF64: fcmpu
; VSX: addis
; VSX: lfs
; VSX: fcmpu
; SPE: efscmpeq
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t1c(float %a) nounwind {
entry:
; ELF64-LABEL: @t1c
; SPE-LABEL: @t1c
; VSX-LABEL: @t1c
  %cmp = fcmp oeq float -0.000000e+00, %a
; ELF64: addis
; ELF64: lfs
; ELF64: fcmpu
; VSX: addis
; VSX: lfs
; VSX: fcmpu
; SPE: efscmpeq
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t2a(double %a) nounwind {
entry:
; ELF64-LABEL: @t2a
; SPE-LABEL: @t2a
; VSX-LABEL: @t2a
  %cmp = fcmp oeq double %a, 0.000000e+00
; ELF64: addis
; ELF64: lfd
; ELF64: fcmpu
; VSX: addis
; VSX: lfd
; VSX: xscmpudp
; SPE: efdcmpeq
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t2b(double %a) nounwind {
entry:
; ELF64-LABEL: @t2b
; SPE-LABEL: @t2b
; VSX-LABEL: @t2b
  %cmp = fcmp oeq double %a, -0.000000e+00
; ELF64: addis
; ELF64: lfd
; ELF64: fcmpu
; VSX: addis
; VSX: lfd
; VSX: xscmpudp
; SPE: efdcmpeq
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t2c(double %a) nounwind {
entry:
; ELF64-LABEL: @t2c
; SPE-LABEL: @t2c
; VSX-LABEL: @t2c
  %cmp = fcmp oeq double -0.000000e+00, %a
; ELF64: addis
; ELF64: lfd
; ELF64: fcmpu
; VSX: addis
; VSX: lfd
; VSX: xscmpudp
; SPE: efdcmpeq
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t4(i8 signext %a) nounwind {
entry:
; ELF64-LABEL: @t4
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
; ELF64-LABEL: @t5
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

define void @t5a(i8 zeroext %a) nounwind {
entry:
; ELF64-LABEL: @t5a
  %cmp = icmp eq i8 1, %a
; ELF64: extsb
; ELF64: cmpw
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t6(i16 signext %a) nounwind {
entry:
; ELF64-LABEL: @t6
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
; ELF64-LABEL: @t7
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

define void @t7a(i16 zeroext %a) nounwind {
entry:
; ELF64-LABEL: @t7a
  %cmp = icmp eq i16 1, %a
; ELF64: extsh
; ELF64: cmpw
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t8(i32 %a) nounwind {
entry:
; ELF64-LABEL: @t8
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
; ELF64-LABEL: @t9
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
; ELF64-LABEL: @t10
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
; ELF64-LABEL: @t11
  %cmp = icmp eq i32 %a, 4096
; ELF64: cmpwi
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t11a(i32 %a) nounwind {
entry:
; ELF64-LABEL: @t11a
  %cmp = icmp eq i32 4096, %a
; ELF64: cmpw
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define void @t12(i8 %a) nounwind {
entry:
; ELF64-LABEL: @t12
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
; ELF64-LABEL: @t13
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
; ELF64-LABEL: @t14
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
; ELF64-LABEL: @t15
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
; ELF64-LABEL: @t16
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
; ELF64-LABEL: @t17
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

define void @t17a(i64 %a) nounwind {
entry:
; ELF64-LABEL: @t17a
  %cmp = icmp eq i64 32768, %a
; Extra operand so we don't match on cmpdi.
; ELF64: cmpd {{[0-9]+}}
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

