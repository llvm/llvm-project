; RUN: llc < %s -mtriple=thumbv7-apple-ios

; If converter was being too cute. It look for root BBs (which don't have
; successors) and use inverse depth first search to traverse the BBs. However
; that doesn't work when the CFG has infinite loops. Simply do a linear
; traversal of all BBs work just fine.

; rdar://9344645

%struct.hc = type { i32, i32, i32, i32 }

define i32 @t(i32 %type) optsize {
entry:
  br i1 undef, label %if.then, label %if.else

if.then:
  unreachable

if.else:
  br i1 undef, label %if.then15, label %if.else18

if.then15:
  unreachable

if.else18:
  switch i32 %type, label %if.else173 [
    i32 3, label %if.then115
    i32 1, label %if.then102
  ]

if.then102:
  br i1 undef, label %cond.true10.i, label %t.exit

cond.true10.i:
  br label %t.exit

t.exit:
  unreachable

if.then115:
  br i1 undef, label %if.else163, label %if.else145

if.else145:
  %call150 = call fastcc %struct.hc* @foo(%struct.hc* undef, i32 34865152) optsize
  br label %while.body172

if.else163:
  %call168 = call fastcc %struct.hc* @foo(%struct.hc* undef, i32 34078720) optsize
  br label %while.body172

while.body172:
  br label %while.body172

if.else173:
  ret i32 -1
}

declare hidden fastcc %struct.hc* @foo(%struct.hc* nocapture, i32) nounwind optsize

