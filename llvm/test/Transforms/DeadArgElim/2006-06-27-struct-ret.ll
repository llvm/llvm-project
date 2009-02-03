; RUN: llvm-as < %s | opt -deadargelim -disable-output

define internal void @build_delaunay({ i32 }* sret  %agg.result) {
        ret void
}

define void @test() {
        call void @build_delaunay( { i32 }* sret  null )
        ret void
}

