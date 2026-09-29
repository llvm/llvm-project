; RUN: opt -mtriple hexagon-- -S -passes='loop(hexagon-loop-idiom,loop-deletion),gvn'
; REQUIRES: asserts

; This tests that the HexagonLoopIdiom pass does not mark LCSSA information
; as preserved. The pass calls SimplifyInstruction is a couple of places,
; which can invalidate LCSSA. Specifically, the uses of a LCSSA phi variable
; are replaced by the incoming value.

define hidden void @test() local_unnamed_addr #0 {
entry:
  br label %if.then63

if.then63:
  br i1 undef, label %do.body311, label %if.end375

do.body311:
  br i1 undef, label %do.end318, label %do.body311

do.end318:
  br i1 undef, label %if.end322, label %if.end375

if.end322:
  %sub325 = sub i32 undef, undef
  br i1 undef, label %do.end329, label %do.body311

do.end329:
  %sub325.lcssa = phi i32 [ %sub325, %if.end322 ]
  br label %do.body330

do.body330:
  %row_width.7 = phi i32 [ %sub325.lcssa, %do.end329 ], [ %dec334, %do.body330 ]
  %sp.5 = phi ptr [ undef, %do.end329 ], [ %incdec.ptr331, %do.body330 ]
  %dp.addr.5 = phi ptr [ undef, %do.end329 ], [ %incdec.ptr332, %do.body330 ]
  %0 = load i8, ptr %sp.5, align 1
  store i8 %0, ptr %dp.addr.5, align 1
  %incdec.ptr332 = getelementptr inbounds i8, ptr %dp.addr.5, i32 1
  %incdec.ptr331 = getelementptr inbounds i8, ptr %sp.5, i32 1
  %dec334 = add i32 %row_width.7, -1
  %cmp335 = icmp eq i32 %dec334, 0
  br i1 %cmp335, label %if.end375, label %do.body330

if.end375:
  ret void
}

attributes #0 = { nounwind }
