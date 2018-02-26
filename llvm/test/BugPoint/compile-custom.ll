; RUN: bugpoint -load %llvmshlibdir/BugpointPasses%shlibext --compile-custom --compile-command="%python %/s.py arg1 arg2" --opt-command opt --output-prefix %t %s | FileCheck %s
; REQUIRES: loadable_module

; Test that arguments are correctly passed in --compile-command.  The output
; of bugpoint includes the output of the custom tool, so we just echo the args
; in the tool and check here.

; CHECK: Error: arg1 arg2

define void @noop() {
    ret void
}
