; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

define void @f1() {
entry:
        store i8 0, ptr null
        ret void
}

; CHECK: strb

define void @f2() {
entry:
        store i16 0, ptr null
        ret void
}

; CHECK: strh

