; RUN: opt -S -passes='default<O1>' -attributor-enable=cgscc-light -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=CGSCCLIGHT %s
; RUN: opt -S -passes='default<O1>' -attributor-enable=module-light -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=MODULELIGHT %s
; RUN: opt -S -passes='default<O1>' -attributor-enable=light -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=LIGHT %s

; RUN: opt -S -passes='default<O1>' -attributor-enable=cgscc -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=CGSCC %s
; RUN: opt -S -passes='default<O1>' -attributor-enable=module -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=MODULE %s
; RUN: opt -S -passes='default<O1>' -attributor-enable=all -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=ALL %s

; CGSCCLIGHT: attributor-light-cgscc,function-attrs
; MODULELIGHT: openmp-opt,attributor-light,ipsccp

; LIGHT: openmp-opt,attributor-light,
; LIGHT-SAME: attributor-light-cgscc,function-attrs


; MODULE: ,openmp-opt,attributor,ipsccp

; CGSCC: inline,attributor-cgscc,function-attrs

; ALL: openmp-opt,attributor,
; ALL-SAME: attributor-cgscc,function-attrs
define ptr @return_arg(ptr %arg) {
  ret ptr %arg
}
