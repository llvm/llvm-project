; RUN: opt -passes="print<cost-model>" 2>&1 -disable-output -mtriple thumbv8.1-m.main -mattr=+mve,+dsp  < %s | FileCheck %s
define i64 @test(i16 %a, i16 %b) {
; CHECK-LABEL: 'test'
; CHECK:  Cost Model: Found an estimated cost of 0 for instruction: %m = mul i32 %as, %bs
;
    %as = sext i16 %a to i32
    %bs = sext i16 %b to i32
    %m = mul i32 %as, %bs
    %ms = sext i32 %m to i64
    ret i64 %ms
}

define i64 @withadd(i16 %a, i16 %b, i64 %c) {
; CHECK-LABEL: 'withadd'
; CHECK:  Cost Model: Found an estimated cost of 0 for instruction: %m = mul i32 %as, %bs
;
    %as = sext i16 %a to i32
    %bs = sext i16 %b to i32
    %m = mul i32 %as, %bs
    %ms = sext i32 %m to i64
    %r = add i64 %c, %ms
    ret i64 %r
}

define i64 @withloads(ptr %pa, ptr %pb, i64 %c) {
; CHECK-LABEL: 'withloads'
; CHECK:  Cost Model: Found an estimated cost of 0 for instruction: %m = mul i32 %as, %bs
;
    %a = load i16, ptr %pa
    %b = load i16, ptr %pb
    %as = sext i16 %a to i32
    %bs = sext i16 %b to i32
    %m = mul i32 %as, %bs
    %ms = sext i32 %m to i64
    %r = add i64 %c, %ms
    ret i64 %r
}
