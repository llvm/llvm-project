; RUN: opt < %s -domtree -gvn -domtree -constmerge -disable-output

define i32 @test1() {
       unreachable
}
