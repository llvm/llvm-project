; RUN: opt -S <%s -simplifycfg | FileCheck %s

define void @test_br(i32 %x) {
entry:
; CHECK: @test_br
; CHECK-NEXT: entry:
; CHECK-NEXT: ret void
  %cmp = icmp eq i32 %x, 10
  br i1 %cmp, label %if.then, label %if.then

if.then:                                          ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

define void @test_switch(i32 %x) nounwind {
entry:
; CHECK: @test_switch
; CHECK-NEXT: entry:
; CHECK-NEXT: ret void
  %rem = srem i32 %x, 3
  switch i32 %rem, label %sw.bb [
    i32 1, label %sw.bb
    i32 10, label %sw.bb
  ]

sw.bb:                                            ; preds = %sw.default, %entry, %entry
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.bb
  ret void
}

define void @test_indirectbr(i32 %x) {
entry:
; CHECK: @test_indirectbr
; CHECK-NEXT: entry:
; Ideally this should now check:
;   CHK-NEXT: ret void
; But that doesn't happen yet. Instead:
; CHECK-NEXT: br label %L1

  %label = bitcast i8* blockaddress(@test_indirectbr, %L1) to i8*
  indirectbr i8* %label, [label %L1, label %L2]

L1:                                               ; preds = %entry
  ret void
L2:                                               ; preds = %entry
  ret void
}
