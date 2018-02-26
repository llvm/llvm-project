; RUN: opt < %s -simplifycfg -S | FileCheck -enable-var-scope %s

; Test basic folding to a conditional branch.
define i32 @foo(i64 %x, i64 %y) nounwind {
; CHECK-LABEL: @foo(
entry:
    %eq = icmp eq i64 %x, %y
    br i1 %eq, label %b, label %switch
switch:
    %lt = icmp slt i64 %x, %y
; CHECK: br i1 %lt, label %a, label %b
    %qux = select i1 %lt, i32 0, i32 2
    switch i32 %qux, label %bees [
        i32 0, label %a
        i32 1, label %b
        i32 2, label %b
    ]
a:
    tail call void @bees.a() nounwind
    ret i32 1
; CHECK: b:
; CHECK-NEXT: %retval = phi i32 [ 0, %switch ], [ 2, %entry ]
b:
    %retval = phi i32 [0, %switch], [0, %switch], [2, %entry]
    tail call void @bees.b() nounwind
    ret i32 %retval
; CHECK-NOT: bees:
bees:
    tail call void @llvm.trap() nounwind
    unreachable
}

; Test basic folding to an unconditional branch.
define i32 @bar(i64 %x, i64 %y) nounwind {
; CHECK-LABEL: @bar(
entry:
; CHECK-NEXT: entry:
; CHECK-NEXT: tail call void @bees.a() [[$NUW:#[0-9]+]]
; CHECK-NEXT: ret i32 0
    %lt = icmp slt i64 %x, %y
    %qux = select i1 %lt, i32 0, i32 2
    switch i32 %qux, label %bees [
        i32 0, label %a
        i32 1, label %b
        i32 2, label %a
    ]
a:
    %retval = phi i32 [0, %entry], [0, %entry], [1, %b]
    tail call void @bees.a() nounwind
    ret i32 0
b:
    tail call void @bees.b() nounwind
    br label %a
bees:
    tail call void @llvm.trap() nounwind
    unreachable
}

; Test the edge case where both values from the select are the default case.
define void @bazz(i64 %x, i64 %y) nounwind {
; CHECK-LABEL: @bazz(
entry:
; CHECK-NEXT: entry:
; CHECK-NEXT: tail call void @bees.b() [[$NUW]]
; CHECK-NEXT: ret void
    %lt = icmp slt i64 %x, %y
    %qux = select i1 %lt, i32 10, i32 12
    switch i32 %qux, label %b [
        i32 0, label %a
        i32 1, label %bees
        i32 2, label %bees
    ]
a:
    tail call void @bees.a() nounwind
    ret void
b:
    tail call void @bees.b() nounwind
    ret void
bees:
    tail call void @llvm.trap()
    unreachable
}

; Test the edge case where both values from the select are equal.
define void @quux(i64 %x, i64 %y) nounwind {
; CHECK-LABEL: @quux(
entry:
; CHECK-NEXT: entry:
; CHECK-NEXT: tail call void @bees.a() [[$NUW]]
; CHECK-NEXT: ret void
    %lt = icmp slt i64 %x, %y
    %qux = select i1 %lt, i32 0, i32 0
    switch i32 %qux, label %b [
        i32 0, label %a
        i32 1, label %bees
        i32 2, label %bees
    ]
a:
    tail call void @bees.a() nounwind
    ret void
b:
    tail call void @bees.b() nounwind
    ret void
bees:
    tail call void @llvm.trap()
    unreachable
}

; A final test, for phi node munging.
define i32 @xyzzy(i64 %x, i64 %y) {
; CHECK-LABEL: @xyzzy(
entry:
    %eq = icmp eq i64 %x, %y
    br i1 %eq, label %r, label %cont
cont:
; CHECK: %lt = icmp slt i64 %x, %y
    %lt = icmp slt i64 %x, %y
; CHECK-NEXT: select i1 %lt, i32 -1, i32 1
    %qux = select i1 %lt, i32 0, i32 2
    switch i32 %qux, label %bees [
        i32 0, label %a
        i32 1, label %r
        i32 2, label %r
    ]
r:
    %val = phi i32 [0, %entry], [1, %cont], [1, %cont]
    ret i32 %val
a:
    ret i32 -1
; CHECK-NOT: bees:
bees:
    tail call void @llvm.trap()
    unreachable
}

declare void @llvm.trap() nounwind noreturn
declare void @bees.a() nounwind
declare void @bees.b() nounwind

; CHECK: attributes [[$NUW]] = { nounwind }
; CHECK: attributes #1 = { noreturn nounwind }
