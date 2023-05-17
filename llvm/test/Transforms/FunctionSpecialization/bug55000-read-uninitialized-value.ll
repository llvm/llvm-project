; RUN: opt -passes="ipsccp<func-spec>" -force-specialization -funcspec-max-iters=2 -funcspec-max-clones=1 -funcspec-for-literal-constant=true -S < %s | FileCheck %s

declare hidden i1 @compare(ptr) align 2
declare hidden { i8, ptr } @getType(ptr) align 2

; CHECK-LABEL: @foo
; CHECK-LABEL: @foo.1
; CHECK-LABEL: @foo.2

define internal void @foo(ptr %TLI, ptr %DL, ptr %Ty, ptr %ValueVTs, ptr %Offsets, i64 %StartingOffset) {
entry:
  %VT = alloca i64, align 8
  br i1 false, label %if.then, label %if.end4

if.then:                                          ; preds = %entry
  ret void

if.end4:                                          ; preds = %entry
  %cmp = call zeroext i1 @compare(ptr undef)
  br i1 %cmp, label %for.body, label %for.cond16

for.body:                                         ; preds = %if.end4
  %add13 = add i64 %StartingOffset, undef
  call void @foo(ptr %TLI, ptr %DL, ptr undef, ptr %ValueVTs, ptr %Offsets, i64 %add13)
  unreachable

for.cond16:                                       ; preds = %for.cond34, %if.end4
  %call27 = call { i8, ptr } @getType(ptr %VT)
  br label %for.cond34

for.cond34:                                       ; preds = %for.body37, %for.cond16
  br i1 undef, label %for.body37, label %for.cond16

for.body37:                                       ; preds = %for.cond34
  %tobool39 = icmp ne ptr %Offsets, null
  br label %for.cond34
}

define hidden { ptr, i32 } @bar(ptr %this) {
entry:
  %Offsets = alloca i64, align 8
  %cmp26 = call zeroext i1 @compare(ptr undef)
  br i1 %cmp26, label %for.body28, label %for.cond.cleanup27

for.cond.cleanup27:                               ; preds = %entry
  ret { ptr, i32 } undef

for.body28:                                       ; preds = %entry
  %call33 = call zeroext i1 @compare(ptr undef)
  br i1 %call33, label %if.then34, label %if.end106

if.then34:                                        ; preds = %for.body28
  call void @foo(ptr %this, ptr undef, ptr undef, ptr undef, ptr null, i64 0)
  unreachable

if.end106:                                        ; preds = %for.body28
  call void @foo(ptr %this, ptr undef, ptr undef, ptr undef, ptr %Offsets, i64 0)
  unreachable
}

