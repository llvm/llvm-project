; RUN: llc < %s -march=arm | grep strb
; RUN: llc < %s -march=arm | grep strh

define void @f1() {
entry:
        store i8 0, i8* null
        ret void
}

define void @f2() {
entry:
        store i16 0, i16* null
        ret void
}
