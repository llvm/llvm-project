; RUN: llc -march=mipsel < %s | FileCheck %s
; RUN: llc -march=mips64el < %s | FileCheck %s

; CHECK-LABEL: test_blez:
; CHECK: blez ${{[0-9]+}}, {{\$|\.L}}BB

define void @test_blez(i32 %a) {
entry:
  %cmp = icmp sgt i32 %a, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @foo1()
  br label %if.end

if.end:
  ret void
}

declare void @foo1()

; CHECK-LABEL: test_bgez:
; CHECK: bgez ${{[0-9]+}}, {{\$|\.L}}BB

define void @test_bgez(i32 %a) {
entry:
  %cmp = icmp slt i32 %a, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @foo1()
  br label %if.end

if.end:
  ret void
}
