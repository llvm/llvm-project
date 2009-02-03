; Linear scan does not currently coalesce any two variables that have
; overlapping live intervals. When two overlapping intervals have the same
; value, they can be joined though.
;
; RUN: llvm-as < %s | llc -march=x86 -regalloc=linearscan | \
; RUN:   not grep {mov %\[A-Z\]\\\{2,3\\\}, %\[A-Z\]\\\{2,3\\\}}

define i64 @test(i64 %x) {
entry:
        %tmp.1 = mul i64 %x, 4294967297         ; <i64> [#uses=1]
        ret i64 %tmp.1
}

