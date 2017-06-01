; RUN: llc -stop-after machine-sink %s -o %t.mir
; RUN: FileCheck %s < %t.mir
; RUN: llc %t.mir -run-pass machine-sink
; Check that branch probabilities are printed in a format that can then be parsed.
; This test fails on powerpc because of an undefined physical register use in the MIR.  See PR31062.
; XFAIL: powerpc

declare void @foo()
declare void @bar()

define void @test(i1 %c) {
; CHECK-LABEL: name: test
entry:
        br i1 %c, label %then, label %else

then:
        call void @foo()
        br label %end
; CHECK: successors: %{{[a-z0-9\-\.]+}}({{0x[0-9a-f]+}}), %{{[a-z0-9\-\.]+}}({{0x[0-9a-f]+}})

else:
        call void @bar()
        br label %end
; CHECK: successors: %{{[a-z0-9\-\.]+}}({{0x[0-9a-f]+}})

end:
        ret void
}
