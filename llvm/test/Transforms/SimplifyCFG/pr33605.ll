; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s

; Skip simplifying unconditional branches from empty blocks in simplifyCFG,
; when it can destroy canonical loop structure.

; void foo();
; bool test(int a, int b, int *c) {
;   bool changed = false;
;   for (unsigned int i = 2; i--;) {
;     int r = a | b;
;     if ( r != c[i]) {
;       c[i] = r;
;       foo();
;       changed = true;
;     }
;   }
;   return changed;
; }

; CHECK-LABEL: @test(
; CHECK: for.cond:
; CHECK-NEXT: %i.0 = phi i32 [ 2, %entry ], [ %dec, %if.end ]
; CHECK: for.body:
; CHECK: br i1 %cmp, label %if.end, label %if.then
; CHECK-NOT: br i1 %cmp, label %for.cond, label %if.then
; CHECK: if.then:
; CHECK: br label %if.end
; CHECK-NOT: br label %for.cond
; CHECK: if.end:
; CHECK: br label %for.cond
define i1 @test(i32 %a, i32 %b, i32* %c) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %if.end, %entry
  %i.0 = phi i32 [ 2, %entry ], [ %dec, %if.end ]
  %changed.0.off0 = phi i1 [ false, %entry ], [ %changed.1.off0, %if.end ]
  %dec = add nsw i32 %i.0, -1
  %tobool = icmp eq i32 %i.0, 0
  br i1 %tobool, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  %changed.0.off0.lcssa = phi i1 [ %changed.0.off0, %for.cond ]
  ret i1 %changed.0.off0.lcssa

for.body:                                         ; preds = %for.cond
  %or = or i32 %a, %b
  %idxprom = sext i32 %dec to i64
  %arrayidx = getelementptr inbounds i32, i32* %c, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %cmp = icmp eq i32 %or, %0
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %for.body
  store i32 %or, i32* %arrayidx, align 4
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %for.body, %if.then
  %changed.1.off0 = phi i1 [ true, %if.then ], [ %changed.0.off0, %for.body ]
  br label %for.cond
}

declare void @foo()
