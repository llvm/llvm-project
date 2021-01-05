; RUN: opt < %s -gvn -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -adce | llvm-dis
; RUN: opt < %s -gvn -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -adce -verify-dom-info | llvm-dis

; This test makes sure that the DominatorTree properly handles
; deletion of edges that go to forward-unreachable regions.
; In this case, %land.end is already forward unreachable when
; the DT gets informed about the deletion of %entry -> %land.end.

@a = common global i32 0, align 4

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  %0 = load i32, i32* @a, align 4
  %cmp = icmp ne i32 %0, 1
  br i1 %cmp, label %land.rhs, label %land.end4

land.rhs:                                         ; preds = %entry
  %1 = load i32, i32* @a, align 4
  %tobool = icmp ne i32 %1, 0
  br i1 %tobool, label %land.rhs1, label %land.end

land.rhs1:                                        ; preds = %land.rhs
  br label %land.end

land.end:                                         ; preds = %land.rhs1, %land.rhs
  %2 = phi i1 [ false, %land.rhs ], [ true, %land.rhs1 ]
  %land.ext = zext i1 %2 to i32
  %conv = trunc i32 %land.ext to i16
  %conv2 = sext i16 %conv to i32
  %tobool3 = icmp ne i32 %conv2, 0
  br label %land.end4

land.end4:                                        ; preds = %land.end, %entry
  %3 = phi i1 [ false, %entry ], [ %tobool3, %land.end ]
  %land.ext5 = zext i1 %3 to i32
  ret i32 0
}
