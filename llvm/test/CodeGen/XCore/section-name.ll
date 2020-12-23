; RUN: llc < %s -march=xcore -o /dev/null 2>&1 | FileCheck %s

@bar = internal global i32 zeroinitializer

define void @".dp.bss"() {
  ret void
}

; CHECK: <unknown>:0: error: invalid symbol redefinition
