; RUN: split-file %s %t
; RUN: not llvm-as < %t/zeroinit-error.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-ZEROINIT %s
; RUN: not llvm-as < %t/global-var.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-GLOBAL-VAR %s
; RUN: not llvm-as < %t/global-array.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-GLOBAL-ARRAY %s
; RUN: not llvm-as < %t/global-struct.ll -o /dev/null 2>&1 | FileCheck --check-prefix=CHECK-GLOBAL-STRUCT %s
; Check target extension type properties are verified in the assembler.

;--- zeroinit-error.ll
define void @foo() {
  %val = freeze target("spirv.DeviceEvent") zeroinitializer
  %val2 = freeze target("unknown_target_type") zeroinitializer
; CHECK-ZEROINIT: error: invalid type for null constant
  ret void
}

;--- global-var.ll
@global_var = external global target("unknown_target_type")
; CHECK-GLOBAL-VAR: Global @global_var has illegal target extension type

;--- global-array.ll
@global_array = external global [4 x target("unknown_target_type")]
; CHECK-GLOBAL-ARRAY: Global @global_array has illegal target extension type

;--- global-struct.ll
@global_struct = external global {target("unknown_target_type")}
; CHECK-GLOBAL-STRUCT: Global @global_struct has illegal target extension type
