; Test for PR463.  This program is erroneous, but should not crash llvm-as.
; RUN: not llvm-as %s -o /dev/null -f |& grep {invalid type for null constant}

@.FOO  = internal global %struct.none zeroinitializer
