; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: invalid value for 'no-jump-tables' attribute: yes

define void @func() #0 {
  ret void
}

attributes #0 = { "no-jump-tables"="yes" }
