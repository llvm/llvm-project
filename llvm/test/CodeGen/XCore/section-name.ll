; RUN: llc < %s -march=xcore

; we used to crash in this.
@bar = internal global i32 zeroinitializer

define void @".dp.bss"() {
  ret void
}
