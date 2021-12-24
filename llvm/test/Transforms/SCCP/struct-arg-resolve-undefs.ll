; RUN: opt -passes=ipsccp -S %s | FileCheck %s

%struct.S = type { i32 }


define void @main() {
; CHECK-LABEL: void @main() {
; CHECK-NEXT:    %r = call i32 @f(%struct.S { i32 100 })
; CHECK-NEXT:    call void @do_report(i32 123)
  %r = call i32 @f(%struct.S { i32 100 })
  call void @do_report(i32 %r)
  ret void
}

declare void @do_report(i32)

define internal i32 @f(%struct.S %s.coerce) {
; CHECK-LABEL: define internal i32 @f(%struct.S %s.coerce)
; CHECK-LABEL: entry:
; CHECK-NEXT:     %call = call i8 @lsh(i8 1, i32 100)
; CHECK-LABEL: if.end:
; CHECK-NEXT:     ret i32 undef
entry:
  %ev = extractvalue %struct.S %s.coerce, 0
  %call = call i8 @lsh(i8 1, i32 %ev)
  %tobool = icmp ne i8 %call, 0
  br i1 %tobool, label %for.cond, label %if.end

for.cond:                                         ; preds = %for.cond, %if.then
  %i.0 = phi i32 [ 0, %entry], [ %inc, %for.cond ]
  %cmp = icmp slt i32 %i.0, 1
  %inc = add nsw i32 %i.0, 1
  br i1 %cmp, label %for.cond, label %if.end

if.end:                                           ; preds = %for.cond, %entry
  ret i32 123
}

define internal i8 @lsh(i8 %l, i32 %r) {
entry:
  %conv = sext i8 %l to i32
  %cmp = icmp slt i32 %conv, 0
  %shr = ashr i32 127, %r
  %cmp4 = icmp sgt i32 %conv, %shr
  %or.cond13 = or i1 %cmp, %cmp4
  %cond = select i1 %or.cond13, i32 %conv, i32 0
  %conv7 = trunc i32 %cond to i8
  ret i8 %conv7
}
