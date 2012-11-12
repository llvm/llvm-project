; RUN: llc < %s

define void @test() {
        %X = alloca {  }                ; <{  }*> [#uses=0]
        ret void
}
