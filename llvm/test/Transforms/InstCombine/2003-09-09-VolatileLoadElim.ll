; RUN: opt < %s -passes=instcombine -S | grep load

define void @test(ptr %P) {
        ; Dead but not deletable!
        %X = load volatile i32, ptr %P              ; <i32> [#uses=0]
        ret void
}
