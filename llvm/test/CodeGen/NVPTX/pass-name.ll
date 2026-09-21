; Check that all legacy PM passes have names
; RUN: llc %s -mtriple=nvptx64 -O3 -enable-new-pm=false --debug-pass=Structure -o /dev/null 2>&1 | FileCheck %s
; CHECK-NOT: Unnamed pass
