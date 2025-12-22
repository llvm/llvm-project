; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: target feature 'foobar' must start with a '+' or '-'
define void @f1() "target-features"="foobar" {
entry:
  ret void
}

; CHECK: target-features attribute should not contain an empty string
define void @f2() "target-features"="+a,-b,,+c" {
entry:
  ret void
}
