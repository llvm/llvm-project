target datalayout = "E-m:a-Fi64-i64:64-n32:64-S128-v256:256:256-v512:512:512"
target triple = "powerpc64-ibm-aix7.2.0.0"

define signext i32 @foo() {
entry:
  ret i32 42
}

