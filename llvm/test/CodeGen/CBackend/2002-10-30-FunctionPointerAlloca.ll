; RUN: llvm-as < %s | llc -march=c

        %BitField = type i32
        %tokenptr = type i32*

define void @test() {
        %pmf1 = alloca %tokenptr (%tokenptr, i8*)*              ; <%tokenptr (%tokenptr, i8*)**> [#uses=0]
        ret void
}

