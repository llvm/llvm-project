; RUN: llc < %s

define void @test() {
        %X = alloca {  }                ; <ptr> [#uses=0]
        ret void
}
