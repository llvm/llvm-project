; Check that all passes have names and block placement is left to ptxas.
; RUN: llc %s -mtriple=nvptx64 -O3 --debug-pass=Structure -o /dev/null 2>&1 | FileCheck %s --implicit-check-not="Branch Probability Basic Block Placement"
; CHECK-NOT: Unnamed pass
