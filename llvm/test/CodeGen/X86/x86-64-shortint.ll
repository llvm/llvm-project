; RUN: llc < %s | FileCheck %s

; CHECK: movswl

target datalayout = "e-p:64:64"
target triple = "x86_64-apple-darwin8"


define void @bar(i16 zeroext  %A) {
        tail call void @foo( i16 signext %A   )
        ret void
}
declare void @foo(i16 signext )

