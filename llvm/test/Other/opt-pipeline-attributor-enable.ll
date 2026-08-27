; RUN: opt -S -passes='default<O1>' -attributor-enable=cgscc-light -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=CGSCCLIGHT %s
; RUN: opt -S -passes='default<O1>' -attributor-enable=module-light -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=MODULELIGHT %s
; RUN: opt -S -passes='default<O1>' -attributor-enable=light -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=LIGHT %s

; RUN: opt -S -passes='default<O1>' -attributor-enable=cgscc -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=CGSCC %s
; RUN: opt -S -passes='default<O1>' -attributor-enable=module -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=MODULE %s
; RUN: opt -S -passes='default<O1>' -attributor-enable=full -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=FULL %s

; CGSCCLIGHT: attributor-light-cgscc,function-attrs
; MODULELIGHT: openmp-opt,attributor-light,ipsccp

; LIGHT: openmp-opt,attributor-light,
; LIGHT-SAME: attributor-light-cgscc,function-attrs


; MODULE: ,openmp-opt,attributor,ipsccp

; CGSCC: inline,attributor-cgscc,function-attrs

; FULL: openmp-opt,attributor,
; FULL-SAME: attributor-cgscc,function-attrs
define ptr @return_arg(ptr %arg) {
  ret ptr %arg
}
