; RUN: split-file %s %t
; RUN: not llvm-as < %t/zeroinit-error.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-ZEROINIT %s
; RUN: not llvm-as < %t/global-var.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-GLOBALVAR %s
; Check target extension type properties are verified in the assembler.

;--- zeroinit-error.ll
define void @foo() {
  %val = freeze target("spirv.DeviceEvent") zeroinitializer
  %val2 = freeze target("unknown_target_type") zeroinitializer
; CHECK-ZEROINIT: error: invalid type for null constant
  ret void
}

;--- global-var.ll
@global = external global target("unknown_target_type")
; CHECK-GLOBALVAR: Global @global has illegal target extension type
