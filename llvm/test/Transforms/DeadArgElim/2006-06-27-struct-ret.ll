; RUN: opt < %s -deadargelim -disable-output

define internal void @build_delaunay({ i32 }* sret({ i32 })  %agg.result) {
        ret void
}

define void @test() {
        call void @build_delaunay({ i32 }* sret({ i32 }) null)
        ret void
}

