; RUN: opt -codegenprepare -mtriple=arm64_32-apple-ios %s -S -o - | FileCheck %s

define void @test_simple_sink(i1* %base, i64 %offset) {
; CHECK-LABEL: @test_simple_sink
; CHECK: next:
; CHECK:   [[BASE8:%.*]] = bitcast i1* %base to i8*
; CHECK:   [[ADDR8:%.*]] = getelementptr i8, i8* [[BASE8]], i64 %offset
; CHECK:   [[ADDR:%.*]] = bitcast i8* [[ADDR8]] to i1*
; CHECK:   load volatile i1, i1* [[ADDR]]
  %addr = getelementptr i1, i1* %base, i64 %offset
  %tst = load i1, i1* %addr
  br i1 %tst, label %next, label %end

next:
  load volatile i1, i1* %addr
  ret void

end:
  ret void
}

define void @test_inbounds_sink(i1* %base, i64 %offset) {
; CHECK-LABEL: @test_inbounds_sink
; CHECK: next:
; CHECK:   [[BASE8:%.*]] = bitcast i1* %base to i8*
; CHECK:   [[ADDR8:%.*]] = getelementptr inbounds i8, i8* [[BASE8]], i64 %offset
; CHECK:   [[ADDR:%.*]] = bitcast i8* [[ADDR8]] to i1*
; CHECK:   load volatile i1, i1* [[ADDR]]
  %addr = getelementptr inbounds i1, i1* %base, i64 %offset
  %tst = load i1, i1* %addr
  br i1 %tst, label %next, label %end

next:
  load volatile i1, i1* %addr
  ret void

end:
  ret void
}

; No address derived via an add can be guaranteed inbounds
define void @test_add_sink(i1* %base, i64 %offset) {
; CHECK-LABEL: @test_add_sink
; CHECK: next:
; CHECK:   [[BASE8:%.*]] = bitcast i1* %base to i8*
; CHECK:   [[ADDR8:%.*]] = getelementptr i8, i8* [[BASE8]], i64 %offset
; CHECK:   [[ADDR:%.*]] = bitcast i8* [[ADDR8]] to i1*
; CHECK:   load volatile i1, i1* [[ADDR]]
  %base64 = ptrtoint i1* %base to i64
  %addr64 = add nsw nuw i64 %base64, %offset
  %addr = inttoptr i64 %addr64 to i1*
  %tst = load i1, i1* %addr
  br i1 %tst, label %next, label %end

next:
  load volatile i1, i1* %addr
  ret void

end:
  ret void
}
