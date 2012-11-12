; Test for PR463.  This program is erroneous, but should not crash llvm-as.
; RUN: not llvm-as %s -o /dev/null 2>&1 | grep "use of undefined type named 'struct.none'"

@.FOO  = internal global %struct.none zeroinitializer
