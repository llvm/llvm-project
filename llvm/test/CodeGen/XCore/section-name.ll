; RUN: not llc < %s -mtriple=xcore -o /dev/null 2>&1 | FileCheck %s

@bar = internal global i32 zeroinitializer

define void @".dp.bss"() {
  ret void
}

; CHECK: <unknown>:0: error: symbol '.dp.bss' is already defined
