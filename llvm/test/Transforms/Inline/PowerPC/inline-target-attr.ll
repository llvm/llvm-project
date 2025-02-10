; RUN: opt < %s -mtriple=powerpc64le-ibm-linux-gnu -S -passes=inline | FileCheck %s
; RUN: opt < %s -mtriple=powerpc64le-ibm-linux-gnu -S -passes='cgscc(inline)' | FileCheck %s

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64le-ibm-linux-gnu"

declare i32 @inlined()

define i32 @foo() #0 {
; CHECK-LABEL: foo
; CHECK: entry
; CHECK-NEXT: call i32 @bar()
; CHECK-NEXT: call i32 @inlined()
entry:
    %1 = call i32 @bar()
    %2 = call i32 @baz()
    %3 = add i32 %1, %2
    ret i32 %3
}

define i32 @bar() #1 {
entry:
    %1 = call i32 @inlined()
    ret i32 %1
}

define i32 @baz() #0 {
entry:
    %1 = call i32 @inlined()
    ret i32 %1
}

attributes #0 = { "target-cpu"="pwr7" "target-features"="+allow-unaligned-fp-access" }
attributes #1 = { "target-cpu"="pwr7" "target-features"="-allow-unaligned-fp-access" }
