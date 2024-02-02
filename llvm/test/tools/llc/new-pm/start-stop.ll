; RUN: llc -mtriple=x86_64-pc-linux-gnu -enable-new-pm -print-pipeline-passes -start-before=mergeicmps -stop-after=gc-lowering -filetype=null %s | FileCheck --match-full-lines %s

; CHECK: function(mergeicmps,expand-memcmp,gc-lowering),machine-function(print)
