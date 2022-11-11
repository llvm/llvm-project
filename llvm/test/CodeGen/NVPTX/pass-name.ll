; Check that all passes have names
; RUN: llc %s -march=nvptx64 -O3 --debug-pass=Structure -o /dev/null 2>&1 | FileCheck %s
; CHECK-NOT: Unnamed pass
