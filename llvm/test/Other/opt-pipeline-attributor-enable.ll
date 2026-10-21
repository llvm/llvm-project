; RUN: opt -S -passes='default<O1>' -attributor-enable=cgscc-light -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=CGSCCLIGHT %s
; RUN: opt -S -passes='default<O1>' -attributor-enable=module-light -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=MODULELIGHT %s
; RUN: opt -S -passes='default<O1>' -attributor-enable=light -print-pipeline-passes %s 2>&1 | FileCheck -check-prefixes=LIGHT,LIGHT-RPO --implicit-check-not=function-attrs --implicit-check-not=attributor %s
; RUN: opt -S -passes='default<O3>' -attributor-enable=light -print-pipeline-passes %s 2>&1 | FileCheck -check-prefixes=LIGHT,LIGHT-RPO --implicit-check-not=function-attrs --implicit-check-not=attributor %s
; RUN: opt -S -passes='thinlto-pre-link<O3>' -attributor-enable=light -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=LIGHT --implicit-check-not=function-attrs --implicit-check-not=attributor %s
; RUN: opt -S -passes='thinlto<O3>' -attributor-enable=light -print-pipeline-passes %s 2>&1 | FileCheck -check-prefixes=LIGHT,LIGHT-RPO --implicit-check-not=function-attrs --implicit-check-not=attributor %s
; RUN: opt -S -passes='lto<O3>' -attributor-enable=light -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=LIGHT-LTO --implicit-check-not=function-attrs --implicit-check-not=attributor %s

; RUN: opt -S -passes='default<O1>' -attributor-enable=cgscc -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=CGSCC %s
; RUN: opt -S -passes='default<O1>' -attributor-enable=module -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=MODULE %s
; RUN: opt -S -passes='default<O1>' -attributor-enable=full -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=FULL %s
; RUN: opt -S -passes='lto<O2>' -attributor-enable=module -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=LTO-MODULE %s
; RUN: opt -S -passes='lto<O2>' -attributor-enable=module-light -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=LTO-MODULE-LIGHT %s
; RUN: opt -S -passes='lto<O2>' -attributor-enable=full -print-pipeline-passes %s 2>&1 | FileCheck -check-prefix=LTO-FULL %s

; CGSCCLIGHT: inline,attributor-light-cgscc,function<
; CGSCCLIGHT-SAME: ),attributor-light-cgscc,function(
; CGSCCLIGHT-SAME: ,elim-avail-extern,rpo-function-attrs,

; MODULELIGHT: openmp-opt,attributor-light,ipsccp
; MODULELIGHT-SAME: inline,function-attrs<skip-non-recursive-function-attrs>,
; MODULELIGHT-SAME: ),function-attrs,function(
; MODULELIGHT-SAME: ,elim-avail-extern,attributor-light,

; LIGHT: openmp-opt,attributor-light,
; LIGHT-SAME: inline,attributor-light-cgscc,
; LIGHT-SAME: ),attributor-light-cgscc,function(
; LIGHT-RPO-SAME: ,elim-avail-extern,attributor-light,

; LIGHT-LTO: ,pgo-icall-prom,cgscc(attributor-light-cgscc,argpromotion,
; LIGHT-LTO-SAME: ,called-value-propagation,attributor-light,globalsplit,
; LIGHT-LTO-SAME: wholeprogramdevirt,attributor-light,norecurse-lto-inference
; LIGHT-LTO-SAME: ,tailcallelim),cgscc(attributor-light-cgscc),

; MODULE: ,openmp-opt,attributor,ipsccp

; CGSCC: inline,attributor-cgscc,function-attrs

; FULL: openmp-opt,attributor,
; FULL-SAME: attributor-cgscc,function-attrs

; LTO-MODULE: wholeprogramdevirt,attributor,norecurse-lto-inference
; LTO-MODULE-LIGHT: wholeprogramdevirt,attributor-light,norecurse-lto-inference
; LTO-FULL: wholeprogramdevirt,attributor,norecurse-lto-inference
define ptr @return_arg(ptr %arg) {
  ret ptr %arg
}
