; RUN: llc < %s -mtriple=i686-pc-win32 | FileCheck %s -check-prefix=FTOL
; RUN: llc < %s -mtriple=i686-pc-mingw32 | FileCheck %s -check-prefix=COMPILERRT
; RUN: llc < %s -mtriple=i686-pc-linux | FileCheck %s -check-prefix=COMPILERRT
; RUN: llc < %s -mtriple=x86_64-pc-win32 | FileCheck %s -check-prefix=COMPILERRT
; RUN: llc < %s -mtriple=x86_64-pc-mingw32 | FileCheck %s -check-prefix=COMPILERRT
; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck %s -check-prefix=COMPILERRT
; RUN: llc < %s -mattr=-sse -O0 -mtriple=i686-pc-win32 | FileCheck %s -check-prefix=FTOL_2

; Win32 targets use the MSVCRT _ftol2 runtime function for fptoui to i64. This
; function has a nonstandard calling convention: the input value is expected on
; the x87 stack instead of the callstack. The input value is popped by the
; callee. Mingw32 uses normal cdecl compiler-rt functions.

define i64 @double_ui64(double %x) nounwind {
entry:
; COMPILERRT: @double_ui64
; COMPILERRT-NOT: calll __ftol2
; FTOL: @double_ui64
; FTOL: fldl
; FTOL: calll __ftol2
; FTOL-NOT: fstp
  %0 = fptoui double %x to i64
  ret i64 %0
}

define i64 @float_ui64(float %x) nounwind {
entry:
; COMPILERRT: @float_ui64
; COMPILERRT-NOT: calll __ftol2
; FTOL: @float_ui64
; FTOL: flds
; FTOL: calll __ftol2
; FTOL-NOT: fstp
  %0 = fptoui float %x to i64
  ret i64 %0
}

define i64 @double_ui64_2(double %x, double %y, double %z) nounwind {
; COMPILERRT: @double_ui64_2
; FTOL: @double_ui64_2
; FTOL_2: @double_ui64_2
;; stack is empty
; FTOL_2: fldl
;; stack is %z
; FTOL_2: fldl
;; stack is %y %z
; FTOL_2: fldl
;; stack is %x %y %z
; FTOL_2: fdiv %st(0), %st(1)
;; stack is %x %1 %z
; FTOL_2: fsubp %st(2)
;; stack is %1 %2
; FTOL_2: fxch
; FTOL_2-NOT: fld
; FTOL_2-NOT: fst
;; stack is %2 %1
; FTOL_2: calll __ftol2
; FTOL_2-NOT: fxch
; FTOL_2-NOT: fld
; FTOL_2-NOT: fst
; FTOL_2: calll __ftol2
;; stack is empty

  %1 = fdiv double %x, %y
  %2 = fsub double %x, %z
  %3 = fptoui double %1 to i64
  %4 = fptoui double %2 to i64
  %5 = sub i64 %3, %4
  ret i64 %5
}

define i64 @double_ui64_3(double %x, double %y, double %z) nounwind {
; COMPILERRT: @double_ui64_3
; FTOL: @double_ui64_3
; FTOL_2: @double_ui64_3
;; stack is empty
; FTOL_2: fldl
;; stack is %z
; FTOL_2: fldl
;; stack is %y %z
; FTOL_2: fldl
;; stack is %x %y %z
; FTOL_2: fdiv %st(0), %st(1)
;; stack is %x %1 %z
; FTOL_2: fsubp %st(2)
;; stack is %1 %2
; FTOL_2-NOT: fxch
; FTOL_2-NOT: fld
; FTOL_2-NOT: fst
;; stack is %1 %2 (still)
; FTOL_2: calll __ftol2
; FTOL_2-NOT: fxch
; FTOL_2-NOT: fld
; FTOL_2-NOT: fst
; FTOL_2: calll __ftol2
;; stack is empty

  %1 = fdiv double %x, %y
  %2 = fsub double %x, %z
  %3 = fptoui double %1 to i64
  %4 = fptoui double %2 to i64
  %5 = sub i64 %4, %3
  ret i64 %5
}

define {double, i64} @double_ui64_4(double %x, double %y) nounwind {
; COMPILERRT: @double_ui64_4
; FTOL: @double_ui64_4
; FTOL_2: @double_ui64_4
;; stack is empty
; FTOL_2: fldl
;; stack is %y
; FTOL_2: fldl
;; stack is %x %y
; FTOL_2: fxch
;; stack is %y %x
; FTOL_2: calll __ftol2
;; stack is %x
; FTOL_2: fld %st(0)
;; stack is %x %x
; FTOL_2: calll __ftol2
;; stack is %x

  %1 = fptoui double %x to i64
  %2 = fptoui double %y to i64
  %3 = sub i64 %1, %2
  %4 = insertvalue {double, i64} undef, double %x, 0
  %5 = insertvalue {double, i64} %4, i64 %3, 1
  ret {double, i64} %5
}

define i32 @double_ui32_5(double %X) {
; FTOL: @double_ui32_5
; FTOL: calll __ftol2
  %tmp.1 = fptoui double %X to i32
  ret i32 %tmp.1
}

define i64 @double_ui64_5(double %X) {
; FTOL: @double_ui64_5
; FTOL: calll __ftol2
  %tmp.1 = fptoui double %X to i64
  ret i64 %tmp.1
}
