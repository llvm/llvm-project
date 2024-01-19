; RUN: llc -mtriple=x86_64-pc-linux-gnu -enable-new-pm -print-pipeline-passes -start-before=mergeicmps -stop-after=gc-lowering -filetype=null %s | FileCheck %s

; CHECK: function(mergeicmps,expand-memcmp,gc-lowering)

