; RUN: opt -S -passes='bdce,instsimplify' < %s | FileCheck %s
; RUN: opt -S -passes=instsimplify < %s | FileCheck %s -check-prefix=CHECK-IO
; RUN: opt -S -passes='debugify,bdce' < %s | FileCheck %s -check-prefix=DEBUGIFY
; RUN: opt -S -passes='debugify,bdce' < %s --try-experimental-debuginfo-iterators | FileCheck %s -check-prefix=DEBUGIFY
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
define signext i32 @bar(i32 signext %x) #0 {
entry:
  %call = tail call signext i32 @foo(i32 signext 5) #0
  %and = and i32 %call, 4
  %or = or i32 %and, %x
  %call1 = tail call signext i32 @foo(i32 signext 3) #0
  %and2 = and i32 %call1, 8
  %or3 = or i32 %or, %and2
  %call4 = tail call signext i32 @foo(i32 signext 2) #0
  %and5 = and i32 %call4, 16
  %or6 = or i32 %or3, %and5
  %call7 = tail call signext i32 @foo(i32 signext 1) #0
  %and8 = and i32 %call7, 32
  %or9 = or i32 %or6, %and8
  %call10 = tail call signext i32 @foo(i32 signext 0) #0
  %and11 = and i32 %call10, 64
  %or12 = or i32 %or9, %and11
  %call13 = tail call signext i32 @foo(i32 signext 4) #0
  %and14 = and i32 %call13, 128
  %or15 = or i32 %or12, %and14
  %shr = ashr i32 %or15, 4
  ret i32 %shr

; CHECK-LABEL: @bar
; CHECK-NOT: tail call signext i32 @foo(i32 signext 5)
; CHECK-NOT: tail call signext i32 @foo(i32 signext 3)
; CHECK: tail call signext i32 @foo(i32 signext 2)
; CHECK: tail call signext i32 @foo(i32 signext 1)
; CHECK: tail call signext i32 @foo(i32 signext 0)
; CHECK: tail call signext i32 @foo(i32 signext 4)
; CHECK: ret i32

; Check that instsimplify is not doing this all on its own.
; CHECK-IO-LABEL: @bar
; CHECK-IO: tail call signext i32 @foo(i32 signext 5)
; CHECK-IO: tail call signext i32 @foo(i32 signext 3)
; CHECK-IO: tail call signext i32 @foo(i32 signext 2)
; CHECK-IO: tail call signext i32 @foo(i32 signext 1)
; CHECK-IO: tail call signext i32 @foo(i32 signext 0)
; CHECK-IO: tail call signext i32 @foo(i32 signext 4)
; CHECK-IO: ret i32
}

; Function Attrs: nounwind readnone
declare signext i32 @foo(i32 signext) #0

; Function Attrs: nounwind readnone
define signext i32 @far(i32 signext %x) #1 {
entry:
  %call = tail call signext i32 @goo(i32 signext 5) #1
  %and = and i32 %call, 4
  %or = or i32 %and, %x
  %call1 = tail call signext i32 @goo(i32 signext 3) #1
  %and2 = and i32 %call1, 8
  %or3 = or i32 %or, %and2
  %call4 = tail call signext i32 @goo(i32 signext 2) #1
  %and5 = and i32 %call4, 16
  %or6 = or i32 %or3, %and5
  %call7 = tail call signext i32 @goo(i32 signext 1) #1
  %and8 = and i32 %call7, 32
  %or9 = or i32 %or6, %and8
  %call10 = tail call signext i32 @goo(i32 signext 0) #1
  %and11 = and i32 %call10, 64
  %or12 = or i32 %or9, %and11
  %call13 = tail call signext i32 @goo(i32 signext 4) #1
  %and14 = and i32 %call13, 128
  %or15 = or i32 %or12, %and14
  %shr = ashr i32 %or15, 4
  ret i32 %shr

; CHECK-LABEL: @far
; Calls to foo(5) and foo(3) are still there, but their results are not used.
; CHECK: tail call signext i32 @goo(i32 signext 5)
; CHECK-NEXT: tail call signext i32 @goo(i32 signext 3)
; CHECK-NEXT: tail call signext i32 @goo(i32 signext 2)
; CHECK: tail call signext i32 @goo(i32 signext 1)
; CHECK: tail call signext i32 @goo(i32 signext 0)
; CHECK: tail call signext i32 @goo(i32 signext 4)
; CHECK: ret i32

; Check that instsimplify is not doing this all on its own.
; CHECK-IO-LABEL: @far
; CHECK-IO: tail call signext i32 @goo(i32 signext 5)
; CHECK-IO: tail call signext i32 @goo(i32 signext 3)
; CHECK-IO: tail call signext i32 @goo(i32 signext 2)
; CHECK-IO: tail call signext i32 @goo(i32 signext 1)
; CHECK-IO: tail call signext i32 @goo(i32 signext 0)
; CHECK-IO: tail call signext i32 @goo(i32 signext 4)
; CHECK-IO: ret i32
}

declare signext i32 @goo(i32 signext) #1

; Function Attrs: nounwind readnone
define signext i32 @tar1(i32 signext %x) #0 {
entry:
  %call = tail call signext i32 @foo(i32 signext 5) #0
  %and = and i32 %call, 33554432
  %or = or i32 %and, %x
  %call1 = tail call signext i32 @foo(i32 signext 3) #0
  %and2 = and i32 %call1, 67108864
  %or3 = or i32 %or, %and2
  %call4 = tail call signext i32 @foo(i32 signext 2) #0
  %and5 = and i32 %call4, 16
  %or6 = or i32 %or3, %and5
  %call7 = tail call signext i32 @foo(i32 signext 1) #0
  %and8 = and i32 %call7, 32
  %or9 = or i32 %or6, %and8
  %call10 = tail call signext i32 @foo(i32 signext 0) #0
  %and11 = and i32 %call10, 64
  %or12 = or i32 %or9, %and11
  %call13 = tail call signext i32 @foo(i32 signext 4) #0
  %and14 = and i32 %call13, 128
  %or15 = or i32 %or12, %and14
  %bs = tail call i32 @llvm.bswap.i32(i32 %or15) #0
  %shr = ashr i32 %bs, 4
  ret i32 %shr

; CHECK-LABEL: @tar1
; CHECK-NOT: tail call signext i32 @foo(i32 signext 5)
; CHECK-NOT: tail call signext i32 @foo(i32 signext 3)
; CHECK: tail call signext i32 @foo(i32 signext 2)
; CHECK: tail call signext i32 @foo(i32 signext 1)
; CHECK: tail call signext i32 @foo(i32 signext 0)
; CHECK: tail call signext i32 @foo(i32 signext 4)
; CHECK: ret i32
}

; Function Attrs: nounwind readnone
declare i32 @llvm.bswap.i32(i32) #0

; Function Attrs: nounwind readnone
define signext i32 @tim(i32 signext %x) #0 {
entry:
  %call = tail call signext i32 @foo(i32 signext 5) #0
  %and = and i32 %call, 536870912
  %or = or i32 %and, %x
  %call1 = tail call signext i32 @foo(i32 signext 3) #0
  %and2 = and i32 %call1, 1073741824
  %or3 = or i32 %or, %and2
  %call4 = tail call signext i32 @foo(i32 signext 2) #0
  %and5 = and i32 %call4, 16
  %or6 = or i32 %or3, %and5
  %call7 = tail call signext i32 @foo(i32 signext 1) #0
  %and8 = and i32 %call7, 32
  %or9 = or i32 %or6, %and8
  %call10 = tail call signext i32 @foo(i32 signext 0) #0
  %and11 = and i32 %call10, 64
  %or12 = or i32 %or9, %and11
  %call13 = tail call signext i32 @foo(i32 signext 4) #0
  %and14 = and i32 %call13, 128
  %or15 = or i32 %or12, %and14
  %bs = tail call i32 @llvm.bitreverse.i32(i32 %or15) #0
  %shr = ashr i32 %bs, 4
  ret i32 %shr

; CHECK-LABEL: @tim
; CHECK-NOT: tail call signext i32 @foo(i32 signext 5)
; CHECK-NOT: tail call signext i32 @foo(i32 signext 3)
; CHECK: tail call signext i32 @foo(i32 signext 2)
; CHECK: tail call signext i32 @foo(i32 signext 1)
; CHECK: tail call signext i32 @foo(i32 signext 0)
; CHECK: tail call signext i32 @foo(i32 signext 4)
; CHECK: ret i32
}

; Function Attrs: nounwind readnone
declare i32 @llvm.bitreverse.i32(i32) #0

; Function Attrs: nounwind readnone
define signext i32 @tar2(i32 signext %x) #0 {
entry:
  %call = tail call signext i32 @foo(i32 signext 5) #0
  %and = and i32 %call, 33554432
  %or = or i32 %and, %x
  %call1 = tail call signext i32 @foo(i32 signext 3) #0
  %and2 = and i32 %call1, 67108864
  %or3 = or i32 %or, %and2
  %call4 = tail call signext i32 @foo(i32 signext 2) #0
  %and5 = and i32 %call4, 16
  %or6 = or i32 %or3, %and5
  %call7 = tail call signext i32 @foo(i32 signext 1) #0
  %and8 = and i32 %call7, 32
  %or9 = or i32 %or6, %and8
  %call10 = tail call signext i32 @foo(i32 signext 0) #0
  %and11 = and i32 %call10, 64
  %or12 = or i32 %or9, %and11
  %call13 = tail call signext i32 @foo(i32 signext 4) #0
  %and14 = and i32 %call13, 128
  %or15 = or i32 %or12, %and14
  %shl = shl i32 %or15, 10
  ret i32 %shl

; CHECK-LABEL: @tar2
; CHECK-NOT: tail call signext i32 @foo(i32 signext 5)
; CHECK-NOT: tail call signext i32 @foo(i32 signext 3)
; CHECK: tail call signext i32 @foo(i32 signext 2)
; CHECK: tail call signext i32 @foo(i32 signext 1)
; CHECK: tail call signext i32 @foo(i32 signext 0)
; CHECK: tail call signext i32 @foo(i32 signext 4)
; CHECK: ret i32
}

; Function Attrs: nounwind readnone
define signext i32 @tar3(i32 signext %x) #0 {
entry:
  %call = tail call signext i32 @foo(i32 signext 5) #0
  %and = and i32 %call, 33554432
  %or = or i32 %and, %x
  %call1 = tail call signext i32 @foo(i32 signext 3) #0
  %and2 = and i32 %call1, 67108864
  %or3 = or i32 %or, %and2
  %call4 = tail call signext i32 @foo(i32 signext 2) #0
  %and5 = and i32 %call4, 16
  %or6 = or i32 %or3, %and5
  %call7 = tail call signext i32 @foo(i32 signext 1) #0
  %and8 = and i32 %call7, 32
  %or9 = or i32 %or6, %and8
  %call10 = tail call signext i32 @foo(i32 signext 0) #0
  %and11 = and i32 %call10, 64
  %or12 = or i32 %or9, %and11
  %call13 = tail call signext i32 @foo(i32 signext 4) #0
  %and14 = and i32 %call13, 128
  %or15 = or i32 %or12, %and14
  %add = add i32 %or15, 5
  %shl = shl i32 %add, 10
  ret i32 %shl

; CHECK-LABEL: @tar3
; CHECK-NOT: tail call signext i32 @foo(i32 signext 5)
; CHECK-NOT: tail call signext i32 @foo(i32 signext 3)
; CHECK: tail call signext i32 @foo(i32 signext 2)
; CHECK: tail call signext i32 @foo(i32 signext 1)
; CHECK: tail call signext i32 @foo(i32 signext 0)
; CHECK: tail call signext i32 @foo(i32 signext 4)
; CHECK: ret i32
}

; Function Attrs: nounwind readnone
define signext i32 @tar4(i32 signext %x) #0 {
entry:
  %call = tail call signext i32 @foo(i32 signext 5) #0
  %and = and i32 %call, 33554432
  %or = or i32 %and, %x
  %call1 = tail call signext i32 @foo(i32 signext 3) #0
  %and2 = and i32 %call1, 67108864
  %or3 = or i32 %or, %and2
  %call4 = tail call signext i32 @foo(i32 signext 2) #0
  %and5 = and i32 %call4, 16
  %or6 = or i32 %or3, %and5
  %call7 = tail call signext i32 @foo(i32 signext 1) #0
  %and8 = and i32 %call7, 32
  %or9 = or i32 %or6, %and8
  %call10 = tail call signext i32 @foo(i32 signext 0) #0
  %and11 = and i32 %call10, 64
  %or12 = or i32 %or9, %and11
  %call13 = tail call signext i32 @foo(i32 signext 4) #0
  %and14 = and i32 %call13, 128
  %or15 = or i32 %or12, %and14
  %sub = sub i32 %or15, 5
  %shl = shl i32 %sub, 10
  ret i32 %shl

; CHECK-LABEL: @tar4
; CHECK-NOT: tail call signext i32 @foo(i32 signext 5)
; CHECK-NOT: tail call signext i32 @foo(i32 signext 3)
; CHECK: tail call signext i32 @foo(i32 signext 2)
; CHECK: tail call signext i32 @foo(i32 signext 1)
; CHECK: tail call signext i32 @foo(i32 signext 0)
; CHECK: tail call signext i32 @foo(i32 signext 4)
; CHECK: ret i32
}

; Function Attrs: nounwind readnone
define signext i32 @tar5(i32 signext %x) #0 {
entry:
  %call = tail call signext i32 @foo(i32 signext 5) #0
  %and = and i32 %call, 33554432
  %or = or i32 %and, %x
  %call1 = tail call signext i32 @foo(i32 signext 3) #0
  %and2 = and i32 %call1, 67108864
  %or3 = or i32 %or, %and2
  %call4 = tail call signext i32 @foo(i32 signext 2) #0
  %and5 = and i32 %call4, 16
  %or6 = or i32 %or3, %and5
  %call7 = tail call signext i32 @foo(i32 signext 1) #0
  %and8 = and i32 %call7, 32
  %or9 = or i32 %or6, %and8
  %call10 = tail call signext i32 @foo(i32 signext 0) #0
  %and11 = and i32 %call10, 64
  %or12 = or i32 %or9, %and11
  %call13 = tail call signext i32 @foo(i32 signext 4) #0
  %and14 = and i32 %call13, 128
  %or15 = or i32 %or12, %and14
  %xor = xor i32 %or15, 5
  %shl = shl i32 %xor, 10
  ret i32 %shl

; CHECK-LABEL: @tar5
; CHECK-NOT: tail call signext i32 @foo(i32 signext 5)
; CHECK-NOT: tail call signext i32 @foo(i32 signext 3)
; CHECK: tail call signext i32 @foo(i32 signext 2)
; CHECK: tail call signext i32 @foo(i32 signext 1)
; CHECK: tail call signext i32 @foo(i32 signext 0)
; CHECK: tail call signext i32 @foo(i32 signext 4)
; CHECK: ret i32
}

; Function Attrs: nounwind readnone
define signext i32 @tar7(i32 signext %x, i1 %b) #0 {
entry:
  %call = tail call signext i32 @foo(i32 signext 5) #0
  %and = and i32 %call, 33554432
  %or = or i32 %and, %x
  %call1 = tail call signext i32 @foo(i32 signext 3) #0
  %and2 = and i32 %call1, 67108864
  %or3 = or i32 %or, %and2
  %call4 = tail call signext i32 @foo(i32 signext 2) #0
  %and5 = and i32 %call4, 16
  %or6 = or i32 %or3, %and5
  %call7 = tail call signext i32 @foo(i32 signext 1) #0
  %and8 = and i32 %call7, 32
  %or9 = or i32 %or6, %and8
  %call10 = tail call signext i32 @foo(i32 signext 0) #0
  %and11 = and i32 %call10, 64
  %or12 = or i32 %or9, %and11
  %call13 = tail call signext i32 @foo(i32 signext 4) #0
  %and14 = and i32 %call13, 128
  %or15 = or i32 %or12, %and14
  %v = select i1 %b, i32 %or15, i32 5
  %shl = shl i32 %v, 10
  ret i32 %shl

; CHECK-LABEL: @tar7
; CHECK-NOT: tail call signext i32 @foo(i32 signext 5)
; CHECK-NOT: tail call signext i32 @foo(i32 signext 3)
; CHECK: tail call signext i32 @foo(i32 signext 2)
; CHECK: tail call signext i32 @foo(i32 signext 1)
; CHECK: tail call signext i32 @foo(i32 signext 0)
; CHECK: tail call signext i32 @foo(i32 signext 4)
; CHECK: ret i32
}

; Function Attrs: nounwind readnone
define signext i16 @tar8(i32 signext %x) #0 {
entry:
  %call = tail call signext i32 @foo(i32 signext 5) #0
  %and = and i32 %call, 33554432
  %or = or i32 %and, %x
  %call1 = tail call signext i32 @foo(i32 signext 3) #0
  %and2 = and i32 %call1, 67108864
  %or3 = or i32 %or, %and2
  %call4 = tail call signext i32 @foo(i32 signext 2) #0
  %and5 = and i32 %call4, 16
  %or6 = or i32 %or3, %and5
  %call7 = tail call signext i32 @foo(i32 signext 1) #0
  %and8 = and i32 %call7, 32
  %or9 = or i32 %or6, %and8
  %call10 = tail call signext i32 @foo(i32 signext 0) #0
  %and11 = and i32 %call10, 64
  %or12 = or i32 %or9, %and11
  %call13 = tail call signext i32 @foo(i32 signext 4) #0
  %and14 = and i32 %call13, 128
  %or15 = or i32 %or12, %and14
  %tr = trunc i32 %or15 to i16
  ret i16 %tr

; CHECK-LABEL: @tar8
; CHECK-NOT: tail call signext i32 @foo(i32 signext 5)
; CHECK-NOT: tail call signext i32 @foo(i32 signext 3)
; CHECK: tail call signext i32 @foo(i32 signext 2)
; CHECK: tail call signext i32 @foo(i32 signext 1)
; CHECK: tail call signext i32 @foo(i32 signext 0)
; CHECK: tail call signext i32 @foo(i32 signext 4)
; CHECK: ret i16
}

; DEBUGIFY-LABEL: @tar9
define signext i16 @tar9(i32 signext %x) #0 {
entry:
  %call = tail call signext i32 @foo(i32 signext 5) #0
  %and = and i32 %call, 33554432
; DEBUGIFY: #dbg_value(i32 %call, {{.*}}, !DIExpression(DW_OP_constu, 33554432, DW_OP_and, DW_OP_stack_value),
  %cast = trunc i32 %call to i16
  ret i16 %cast
}

attributes #0 = { nounwind readnone willreturn }
attributes #1 = { nounwind }

