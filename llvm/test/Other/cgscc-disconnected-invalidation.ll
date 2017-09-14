; Test that patterns of transformations which disconnect a region of the call
; graph mid-traversal and then invalidate it function correctly.
;
; RUN: opt -S -passes='cgscc(inline,function(simplify-cfg))' < %s | FileCheck %s

define internal void @test_scc_internal(i1 %flag) {
; CHECK-NOT: @test_scc_internal
entry:
  br i1 %flag, label %then, label %else

then:
  call void @test_scc_internal(i1 false)
  call void @test_scc_external()
  br label %else

else:
  ret void
}

define void @test_scc_external() {
; CHECK-LABEL: define void @test_scc_external()
entry:
  call void @test_scc_internal(i1 false)
  ret void
}

define internal void @test_refscc_internal(i1 %flag, i8* %ptr) {
; CHECK-NOT: @test_refscc_internal
entry:
  br i1 %flag, label %then, label %else

then:
  call void @test_refscc_internal(i1 false, i8* bitcast (i8* ()* @test_refscc_external to i8*))
  br label %else

else:
  ret void
}

define i8* @test_refscc_external() {
; CHECK-LABEL: define i8* @test_refscc_external()
entry:
  br i1 true, label %then, label %else
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i8* null
; CHECK-NEXT:  }
; CHECK-NOT: @test_refscc_internal

then:
  ret i8* null

else:
  ret i8* bitcast (void (i1, i8*)* @test_refscc_internal to i8*)
}
