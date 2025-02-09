; RUN: not llc -mtriple=x86_64-pc-linux-gnu -passes=foo -start-before=mergeicmps -stop-after=gc-lowering -filetype=null %s 2>&1 | FileCheck  %s

; CHECK: warning: --passes cannot be used with start-before and stop-after.
