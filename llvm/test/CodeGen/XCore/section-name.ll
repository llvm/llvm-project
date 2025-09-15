; RUN: llc < %s -mtriple=xcore | FileCheck %s

@bar = internal global i32 zeroinitializer

define void @".dp.bss"() {
  ret void
}

; CHECK: .dp.bss:
