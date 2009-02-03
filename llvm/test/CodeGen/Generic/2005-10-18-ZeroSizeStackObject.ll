; RUN: llvm-as < %s | llc

define void @test() {
        %X = alloca {  }                ; <{  }*> [#uses=0]
        ret void
}
