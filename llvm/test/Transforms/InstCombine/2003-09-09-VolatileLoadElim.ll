; RUN: opt < %s -instcombine -S | grep load

define void @test(i32* %P) {
        ; Dead but not deletable!
        %X = load volatile i32* %P              ; <i32> [#uses=0]
        ret void
}
