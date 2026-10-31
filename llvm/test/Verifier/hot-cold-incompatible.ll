; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: Attributes 'hot' and 'cold' are incompatible
define void @_62_16_3() #0 {
  ret void
}

attributes #0 = { hot cold }
