; REQUIRES: plugins, examples
; UNSUPPORTED: target={{.*windows.*}}

; RUN: not opt < %s -load-pass-plugin=%t/nonexistant.so -disable-output 2>&1 | FileCheck %s
; CHECK: Could not load library {{.*}}nonexistant.so
