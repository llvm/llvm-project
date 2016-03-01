; RUN: llc -march=mipsel < %s | \
; RUN:    FileCheck %s -check-prefix=ALL
; RUN: llc -march=mips64el < %s | \
; RUN:    FileCheck %s -check-prefix=ALL -check-prefix=GP64

declare void @foo1()

define void @test_blez(i32 %a) {
; ALL-LABEL: test_blez:
; ALL: blez ${{[0-9]+}}, $BB
entry:
  %cmp = icmp sgt i32 %a, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @foo1()
  br label %if.end

if.end:
  ret void
}

define void @test_bgez(i32 %a) {
entry:
; ALL-LABEL: test_bgez:
; ALL: bgez ${{[0-9]+}}, $BB
  %cmp = icmp slt i32 %a, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @foo1()
  br label %if.end

if.end:
  ret void
}

define void @test_blez_64(i64 %a) {
; GP64-LABEL: test_blez_64:
; GP64: blez ${{[0-9]+}}, $BB
entry:
  %cmp = icmp sgt i64 %a, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @foo1()
  br label %if.end

if.end:
  ret void
}

define void @test_bgez_64(i64 %a) {
entry:
; ALL-LABEL: test_bgez_64:
; ALL: bgez ${{[0-9]+}}, $BB
  %cmp = icmp slt i64 %a, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @foo1()
  br label %if.end

if.end:
  ret void
}
