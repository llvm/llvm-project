; RUN: llc < %s -mtriple=i386-apple-darwin10
; PR4910

%0 = type { i32, i32, i32, i32 }

@boot_cpu_id = external global i32                ; <ptr> [#uses=1]
@cpu_logical = common global i32 0, align 4       ; <ptr> [#uses=1]

define void @topo_probe_0xb() nounwind ssp {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc38, %entry
  %0 = phi i32 [ 0, %entry ], [ %inc40, %for.inc38 ] ; <i32> [#uses=3]
  %cmp = icmp slt i32 %0, 3                       ; <i1> [#uses=1]
  br i1 %cmp, label %for.body, label %for.end41

for.body:                                         ; preds = %for.cond
  %1 = tail call %0 asm sideeffect "cpuid", "={ax},={bx},={cx},={dx},0,{cx},~{dirflag},~{fpsr},~{flags}"(i32 11, i32 %0) nounwind ; <%0> [#uses=3]
  %asmresult.i = extractvalue %0 %1, 0            ; <i32> [#uses=1]
  %asmresult10.i = extractvalue %0 %1, 2          ; <i32> [#uses=1]
  %and = and i32 %asmresult.i, 31                 ; <i32> [#uses=2]
  %shr42 = lshr i32 %asmresult10.i, 8             ; <i32> [#uses=1]
  %and12 = and i32 %shr42, 255                    ; <i32> [#uses=2]
  %cmp14 = icmp eq i32 %and12, 0                  ; <i1> [#uses=1]
  br i1 %cmp14, label %for.end41, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %for.body
  %asmresult9.i = extractvalue %0 %1, 1           ; <i32> [#uses=1]
  %and7 = and i32 %asmresult9.i, 65535            ; <i32> [#uses=1]
  %cmp16 = icmp eq i32 %and7, 0                   ; <i1> [#uses=1]
  br i1 %cmp16, label %for.end41, label %for.cond17.preheader

for.cond17.preheader:                             ; preds = %lor.lhs.false
  %tmp24 = load i32, ptr @boot_cpu_id                 ; <i32> [#uses=1]
  %shr26 = ashr i32 %tmp24, %and                  ; <i32> [#uses=1]
  br label %for.body20

for.body20:                                       ; preds = %for.body20, %for.cond17.preheader
  %2 = phi i32 [ 0, %for.cond17.preheader ], [ %inc32, %for.body20 ] ; <i32> [#uses=2]
  %cnt.143 = phi i32 [ 0, %for.cond17.preheader ], [ %inc.cnt.1, %for.body20 ] ; <i32> [#uses=1]
  %shr23 = ashr i32 %2, %and                      ; <i32> [#uses=1]
  %cmp27 = icmp eq i32 %shr23, %shr26             ; <i1> [#uses=1]
  %inc = zext i1 %cmp27 to i32                    ; <i32> [#uses=1]
  %inc.cnt.1 = add i32 %inc, %cnt.143             ; <i32> [#uses=2]
  %inc32 = add nsw i32 %2, 1                      ; <i32> [#uses=2]
  %exitcond = icmp eq i32 %inc32, 255             ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body20

for.end:                                          ; preds = %for.body20
  %cmp34 = icmp eq i32 %and12, 1                  ; <i1> [#uses=1]
  br i1 %cmp34, label %if.then35, label %for.inc38

if.then35:                                        ; preds = %for.end
  store i32 %inc.cnt.1, ptr @cpu_logical
  br label %for.inc38

for.inc38:                                        ; preds = %for.end, %if.then35
  %inc40 = add nsw i32 %0, 1                      ; <i32> [#uses=1]
  br label %for.cond

for.end41:                                        ; preds = %lor.lhs.false, %for.body, %for.cond
  ret void
}
