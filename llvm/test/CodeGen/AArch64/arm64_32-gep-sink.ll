; RUN: opt -codegenprepare -mtriple=arm64_32-apple-ios %s -S -o - | FileCheck %s

define void @test_simple_sink(ptr %base, i64 %offset) {
; CHECK-LABEL: @test_simple_sink
; CHECK: next:
; CHECK:   [[ADDR8:%.*]] = getelementptr i8, ptr %base, i64 %offset
; CHECK:   load volatile i1, ptr [[ADDR8]]
  %addr = getelementptr i1, ptr %base, i64 %offset
  %tst = load i1, ptr %addr
  br i1 %tst, label %next, label %end

next:
  load volatile i1, ptr %addr
  ret void

end:
  ret void
}

define void @test_inbounds_sink(ptr %base, i64 %offset) {
; CHECK-LABEL: @test_inbounds_sink
; CHECK: next:
; CHECK:   [[ADDR8:%.*]] = getelementptr inbounds i8, ptr %base, i64 %offset
; CHECK:   load volatile i1, ptr [[ADDR8]]
  %addr = getelementptr inbounds i1, ptr %base, i64 %offset
  %tst = load i1, ptr %addr
  br i1 %tst, label %next, label %end

next:
  load volatile i1, ptr %addr
  ret void

end:
  ret void
}

; No address derived via an add can be guaranteed inbounds
define void @test_add_sink(ptr %base, i64 %offset) {
; CHECK-LABEL: @test_add_sink
; CHECK: next:
; CHECK:   [[ADDR8:%.*]] = getelementptr i8, ptr %base, i64 %offset
; CHECK:   load volatile i1, ptr [[ADDR8]]
  %base64 = ptrtoint ptr %base to i64
  %addr64 = add nsw nuw i64 %base64, %offset
  %addr = inttoptr i64 %addr64 to ptr
  %tst = load i1, ptr %addr
  br i1 %tst, label %next, label %end

next:
  load volatile i1, ptr %addr
  ret void

end:
  ret void
}
