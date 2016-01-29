; FIXME: FastISel currently returns false if it hits code that uses VSX
; registers and with -fast-isel-abort=1 turned on the test case will then fail.
; When fastisel better supports VSX fix up this test case.
;
; RUN: llc < %s -O0 -verify-machineinstrs -fast-isel-abort=1 -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=-vsx | FileCheck %s --check-prefix=ELF64

define zeroext i1 @rettrue() nounwind {
entry:
; ELF64-LABEL: rettrue
; ELF64: li 3, 1
; ELF64: blr
  ret i1 true
}

define zeroext i1 @retfalse() nounwind {
entry:
; ELF64-LABEL: retfalse
; ELF64: li 3, 0
; ELF64: blr
  ret i1 false
}

define signext i1 @retstrue() nounwind {
entry:
; ELF64-LABEL: retstrue
; ELF64: li 3, -1
; ELF64: blr
  ret i1 true
}

define signext i1 @retsfalse() nounwind {
entry:
; ELF64-LABEL: retsfalse
; ELF64: li 3, 0
; ELF64: blr
  ret i1 false
}

define signext i8 @ret2(i8 signext %a) nounwind {
entry:
; ELF64-LABEL: ret2
; ELF64: extsb
; ELF64: blr
  ret i8 %a
}

define zeroext i8 @ret3(i8 signext %a) nounwind {
entry:
; ELF64-LABEL: ret3
; ELF64: rldicl {{[0-9]+}}, {{[0-9]+}}, 0, 56
; ELF64: blr
  ret i8 %a
}

define signext i16 @ret4(i16 signext %a) nounwind {
entry:
; ELF64-LABEL: ret4
; ELF64: extsh
; ELF64: blr
  ret i16 %a
}

define zeroext i16 @ret5(i16 signext %a) nounwind {
entry:
; ELF64-LABEL: ret5
; ELF64: rldicl {{[0-9]+}}, {{[0-9]+}}, 0, 48
; ELF64: blr
  ret i16 %a
}

define i16 @ret6(i16 %a) nounwind {
entry:
; ELF64-LABEL: ret6
; ELF64: rldicl {{[0-9]+}}, {{[0-9]+}}, 0, 48
; ELF64: blr
  ret i16 %a
}

define signext i32 @ret7(i32 signext %a) nounwind {
entry:
; ELF64-LABEL: ret7
; ELF64: extsw
; ELF64: blr
  ret i32 %a
}

define zeroext i32 @ret8(i32 signext %a) nounwind {
entry:
; ELF64-LABEL: ret8
; ELF64: rldicl {{[0-9]+}}, {{[0-9]+}}, 0, 32
; ELF64: blr
  ret i32 %a
}

define i32 @ret9(i32 %a) nounwind {
entry:
; ELF64-LABEL: ret9
; ELF64: rldicl {{[0-9]+}}, {{[0-9]+}}, 0, 32
; ELF64: blr
  ret i32 %a
}

define i64 @ret10(i64 %a) nounwind {
entry:
; ELF64-LABEL: ret10
; ELF64-NOT: exts
; ELF64-NOT: rldicl
; ELF64: blr
  ret i64 %a
}

define float @ret11(float %a) nounwind {
entry:
; ELF64-LABEL: ret11
; ELF64: blr
  ret float %a
}

define double @ret12(double %a) nounwind {
entry:
; ELF64-LABEL: ret12
; ELF64: blr
  ret double %a
}

define i8 @ret13() nounwind {
entry:
; ELF64-LABEL: ret13
; ELF64: li
; ELF64: blr
  ret i8 15;
}

define i16 @ret14() nounwind {
entry:
; ELF64-LABEL: ret14
; ELF64: li
; ELF64: blr
  ret i16 -225;
}

define i32 @ret15() nounwind {
entry:
; ELF64-LABEL: ret15
; ELF64: lis
; ELF64: ori
; ELF64: blr
  ret i32 278135;
}

define i64 @ret16() nounwind {
entry:
; ELF64-LABEL: ret16
; ELF64: li
; ELF64: sldi
; ELF64: oris
; ELF64: ori
; ELF64: blr
  ret i64 27813515225;
}

define float @ret17() nounwind {
entry:
; ELF64-LABEL: ret17
; ELF64: addis
; ELF64: lfs
; ELF64: blr
  ret float 2.5;
}

define double @ret18() nounwind {
entry:
; ELF64-LABEL: ret18
; ELF64: addis
; ELF64: lfd
; ELF64: blr
  ret double 2.5e-33;
}

define zeroext i32 @ret19() nounwind {
entry:
; ELF64-LABEL: ret19
; ELF64: li
; ELF64: oris
; ELF64: ori
; ELF64: blr
  ret i32 -1
}

define zeroext i16 @ret20() nounwind {
entry:
; ELF64-LABEL: ret20
; ELF64: lis{{.*}}0
; ELF64: ori{{.*}}32768
; ELF64: blr
  ret i16 32768
}
