; RUN: opt -S <%s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 | FileCheck %s

define void @test_br(i32 %x) {
entry:
; CHECK-LABEL: @test_br(
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
; CHECK-LABEL: @test_switch(
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
; CHECK-LABEL: @test_indirectbr(
; CHECK-NEXT: entry:
; CHECK-NEXT:   ret void

  %label = bitcast ptr blockaddress(@test_indirectbr, %L1) to ptr
  indirectbr ptr %label, [label %L1, label %L2]

L1:                                               ; preds = %entry
  ret void
L2:                                               ; preds = %entry
  ret void
}
