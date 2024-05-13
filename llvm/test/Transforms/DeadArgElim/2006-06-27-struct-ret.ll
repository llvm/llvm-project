; RUN: opt < %s -passes=deadargelim -disable-output

define internal void @build_delaunay(ptr sret({ i32 })  %agg.result) {
        ret void
}

define void @test() {
        call void @build_delaunay(ptr sret({ i32 }) null)
        ret void
}

