; RUN: opt < %s -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

; CHECK-LABEL: no_hoist
; CHECK: edge %entry -> %if.end probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK: edge %entry -> %if.then probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK: edge %if.end -> %if.end4 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK: edge %if.end -> %if.then3 probability is 0x40000000 / 0x80000000 = 50.00%
define dso_local void @no_hoist(i64 %arg1, i64 %arg2) local_unnamed_addr #0 {
entry:
  %and = and i64 %arg1, 1152921504606846976
  %tobool.not = icmp eq i64 %and, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  tail call void @bar()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %and1 = and i64 %arg2, 1152921504606846976
  %tobool2.not = icmp eq i64 %and1, 0
  br i1 %tobool2.not, label %if.end4, label %if.then3

if.then3:                                         ; preds = %if.end
  tail call void @baz()
  br label %if.end4

if.end4:                                          ; preds = %if.then3, %if.end
  ret void
}

; CHECK-LABEL: hoist
; CHECK: edge %entry -> %if.end probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK: edge %entry -> %if.then probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK: edge %if.end -> %if.end4 probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK: edge %if.end -> %if.then3 probability is 0x40000000 / 0x80000000 = 50.00%
define dso_local void @hoist(i64 %arg1, i64 %arg2) local_unnamed_addr #0 {
entry:
  %const = bitcast i64 1152921504606846976 to i64
  %and = and i64 %arg1, %const
  %tobool.not = icmp eq i64 %and, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  tail call void @bar()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %and1 = and i64 %arg2, %const
  %tobool2.not = icmp eq i64 %and1, 0
  br i1 %tobool2.not, label %if.end4, label %if.then3

if.then3:                                         ; preds = %if.end
  tail call void @baz()
  br label %if.end4

if.end4:                                          ; preds = %if.then3, %if.end
  ret void
}

declare dso_local void @bar() local_unnamed_addr #1

declare dso_local void @baz() local_unnamed_addr #1
