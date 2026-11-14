; Test for PR463.  This program is erroneous, but should not crash llvm-as.
; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s
; CHECK: invalid type for null constant

@.FOO  = internal global %struct.none zeroinitializer
