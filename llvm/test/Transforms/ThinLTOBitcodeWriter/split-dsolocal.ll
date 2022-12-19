; RUN: opt  -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract  -b -n 0 -o %t0.bc %t
; RUN: llvm-modextract  -b -n 1 -o %t1.bc %t
; RUN: llvm-dis  -o - %t0.bc | FileCheck --check-prefix=M0 %s
; RUN: llvm-dis  -o - %t1.bc | FileCheck --check-prefix=M1 %s

; M0: @default = external constant [1 x i8]
; M0: @hidden = external hidden constant [1 x i8]
; M0: @al = external global [1 x i8]

; M1: @default = dso_local constant [1 x i8] c"0", !type !0
; M1: @hidden = hidden constant [1 x i8] c"0", !type !0
; M1: @al = dso_local alias [1 x i8], ptr @default
@default = dso_local constant [1 x i8] c"0", !type !0
@hidden = dso_local hidden constant [1 x i8] c"0", !type !0

@al = dso_local alias [1 x i8], ptr @default

define ptr @f_default() { ret ptr @default }
define ptr @f_hidden() { ret ptr @hidden }
define ptr @f_al() { ret ptr @al }

!0 = !{i32 0, !"typeid"}
