; RUN: llc -mtriple=hexagon -relocation-model=pic < %s | FileCheck %s

; CHECK: r{{[0-9]+}} = add({{pc|PC}},##
; CHECK: r{{[0-9]+}} = memw(r{{[0-9]+}}+r{{[0-9]+}}<<#2)
; CHECK: r{{[0-9]+}} = add(r{{[0-9]+}},r{{[0-9]+}})


define i32 @test(i32 %y) nounwind {
entry:
  switch i32 %y, label %sw.epilog [
    i32 1, label %sw.bb
    i32 2, label %sw.bb1
    i32 3, label %sw.bb2
    i32 4, label %sw.bb3
    i32 5, label %sw.bb4
  ]

sw.bb:                                            ; preds = %entry
  tail call void @baz1() nounwind
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  tail call void @baz2(i32 2, i32 78) nounwind
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  tail call void @baz3(i32 59) nounwind
  br label %sw.epilog

sw.bb3:                                           ; preds = %entry
  tail call void @baz4(i32 4, i32 14) nounwind
  br label %sw.epilog

sw.bb4:                                           ; preds = %entry
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.bb4, %sw.bb3, %sw.bb2, %sw.bb1, %sw.bb, %entry
  %y.addr.0 = phi i32 [ %y, %entry ], [ 14, %sw.bb4 ], [ 4, %sw.bb3 ], [ 3, %sw.bb2 ], [ 2, %sw.bb1 ], [ 1, %sw.bb ]
  ret i32 %y.addr.0
}

declare void @baz1(...)

declare void @baz2(i32, i32)

declare void @baz3(i32)

declare void @baz4(i32, i32)
