; RUN: bugpoint -load %llvmshlibdir/BugpointPasses%pluginext --compile-custom --compile-command="%python %/s.py arg1 arg2" --output-prefix %t %s | FileCheck %s
; REQUIRES: plugins

; Test that arguments are correctly passed in --compile-command.  The output
; of bugpoint includes the output of the custom tool, so we just echo the args
; in the tool and check here.

; CHECK: Error: arg1 arg2

define void @noop() {
    ret void
}
