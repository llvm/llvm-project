; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: invalid value for 'frame-pointer' attribute: arst

define void @func() #0 {
  ret void
}

attributes #0 = { "frame-pointer"="arst" }
