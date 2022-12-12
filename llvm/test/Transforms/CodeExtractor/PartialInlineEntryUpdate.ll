; RUN: opt < %s -skip-partial-inlining-cost-analysis -passes=partial-inliner -S  | FileCheck %s

define i32 @Func(i1 %cond, i32* align 4 %align.val) !prof !1 {
; CHECK: @Func({{.*}}) !prof [[REMAINCOUNT:![0-9]+]]
entry:
  br i1 %cond, label %if.then, label %return
if.then:
  ; Dummy store to have more than 0 uses
  store i32 10, i32* %align.val, align 4
  br label %return
return:             ; preds = %entry
  ret i32 0
}

define internal i32 @Caller1(i1 %cond, i32* align 2 %align.val) !prof !3{
entry:
; CHECK-LABEL: @Caller1
; CHECK: br
; CHECK: call void @Func.1.
; CHECK: br
; CHECK: call void @Func.1.
  %val = call i32 @Func(i1 %cond, i32* %align.val)
  %val2 = call i32 @Func(i1 %cond, i32* %align.val)
  ret i32 %val
}

define internal i32 @Caller2(i1 %cond, i32* align 2 %align.val) !prof !2{
entry:
; CHECK-LABEL: @Caller2
; CHECK: br
; CHECK: call void @Func.1.
  %val = call i32 @Func(i1 %cond, i32* %align.val)
  ret i32 %val
}

; CHECK: [[REMAINCOUNT]] = !{!"function_entry_count", i64 150}
!1 = !{!"function_entry_count", i64 200}
!2 = !{!"function_entry_count", i64 10}
!3 = !{!"function_entry_count", i64 20}

