; Thanks to Daniele Vettorel for providing the original source code
; for this test.
;
; RUN: opt < %s -task-simplify -S -o - | FileCheck %s
; RUN: opt < %s -passes="task-simplify" -S -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline norecurse nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) #0 {
entry:
  %syncreg = call token @llvm.syncregion.start()
  %cmp = icmp sgt i32 %argc, 1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %arrayidx = getelementptr inbounds i8*, i8** %argv, i64 1
  %0 = load i8*, i8** %arrayidx, align 8
  %call = call i32 @atoi(i8* %0) #4
  %conv = sext i32 %call to i64
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %n.0 = phi i64 [ %conv, %if.then ], [ 10, %entry ]
  %cmp1 = icmp sgt i32 %argc, 2
  br i1 %cmp1, label %if.then2, label %if.end6

if.then2:                                         ; preds = %if.end
  %arrayidx3 = getelementptr inbounds i8*, i8** %argv, i64 2
  %1 = load i8*, i8** %arrayidx3, align 8
  %call4 = call i32 @atoi(i8* %1) #4
  %conv5 = sext i32 %call4 to i64
  br label %if.end6

if.end6:                                          ; preds = %if.then2, %if.end
  %k.0 = phi i64 [ %conv5, %if.then2 ], [ 5, %if.end ]
  %cmp7 = icmp sgt i32 %argc, 3
  br i1 %cmp7, label %if.then8, label %if.end12

if.then8:                                         ; preds = %if.end6
  %arrayidx9 = getelementptr inbounds i8*, i8** %argv, i64 3
  %2 = load i8*, i8** %arrayidx9, align 8
  %call10 = call i32 @atoi(i8* %2) #4
  %conv11 = sext i32 %call10 to i64
  br label %if.end12

if.end12:                                         ; preds = %if.then8, %if.end6
  %o.0 = phi i64 [ %conv11, %if.then8 ], [ 1, %if.end6 ]
  br label %for.cond

for.cond:                                         ; preds = %for.inc18, %if.end12
  %j.0 = phi i64 [ 0, %if.end12 ], [ %inc19, %for.inc18 ]
  %cmp13 = icmp ult i64 %j.0, %o.0
  br i1 %cmp13, label %for.cond14, label %for.end20

for.cond14:                                       ; preds = %for.cond, %for.inc
  %i.0 = phi i64 [ %inc, %for.inc ], [ 0, %for.cond ]
  %cmp15 = icmp ult i64 %i.0, %k.0
  br i1 %cmp15, label %for.body16, label %for.inc18

for.body16:                                       ; preds = %for.cond14
  detach within %syncreg, label %det.achd, label %for.inc

det.achd:                                         ; preds = %for.body16
  %call17 = call i64 @_Z12testFunctionm(i64 %n.0)
  reattach within %syncreg, label %for.inc

for.inc:                                          ; preds = %for.body16, %det.achd
  %inc = add i64 %i.0, 1
  br label %for.cond14

for.inc18:                                        ; preds = %for.cond14
  %inc19 = add i64 %j.0, 1
  br label %for.cond

for.end20:                                        ; preds = %for.cond
  sync within %syncreg, label %sync.continue

; Verify that the Task-simplify pass does not remove the sync in this function.
; CHECK: sync within %syncreg

sync.continue:                                    ; preds = %for.end20
  ret i32 0
}

; Function Attrs: nounwind readonly
declare dso_local i32 @atoi(i8*) #1

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #2

declare dso_local i64 @_Z12testFunctionm(i64) #3

attributes #0 = { noinline norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind readonly }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.0.0 (git@github.com:wsmoses/Tapir-Clang.git df694b62fc84b34c487e1abdfc76e010d0117221) (git@github.com:wsmoses/Tapir-LLVM.git 3f61ca59b497e08d8ec2e0d6ca942a3ba4f12ff6)"}
