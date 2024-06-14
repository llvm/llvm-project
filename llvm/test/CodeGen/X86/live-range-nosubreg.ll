; RUN: llc < %s

; This testcase used to crash. See PR29132.

target triple = "x86_64-unknown-linux-gnu"

@a = common local_unnamed_addr global i16 0, align 2
@c = common global i32 0, align 4
@d = common local_unnamed_addr global i8 0, align 1
@b = common global i32 0, align 4

; Function Attrs: norecurse nounwind optsize uwtable
define i32 @main() local_unnamed_addr #0 {
entry:
  %0 = load volatile i32, ptr @c, align 4
  %tobool = icmp eq i32 %0, 0
  %1 = load i16, ptr @a, align 2
  br i1 %tobool, label %lor.rhs, label %lor.end

lor.rhs:                                          ; preds = %entry
  %inc = add i16 %1, 1
  store i16 %inc, ptr @a, align 2
  br label %lor.end

lor.end:                                          ; preds = %entry, %lor.rhs
  %2 = phi i16 [ %inc, %lor.rhs ], [ %1, %entry ]
  %dec = add i16 %2, -1
  store i16 %dec, ptr @a, align 2
  %3 = load i8, ptr @d, align 1
  %sub = sub i8 0, %3
  %tobool4 = icmp eq i16 %dec, 0
  br i1 %tobool4, label %land.end, label %land.rhs

land.rhs:                                         ; preds = %lor.end
  %4 = load volatile i32, ptr @b, align 4
  %tobool5 = icmp ne i32 %4, 0
  br label %land.end

land.end:                                         ; preds = %lor.end, %land.rhs
  %5 = phi i1 [ false, %lor.end ], [ %tobool5, %land.rhs ]
  %land.ext = zext i1 %5 to i8
  %or = or i8 %land.ext, %sub
  store i8 %or, ptr @d, align 1
  ret i32 0
}

attributes #0 = { norecurse nounwind optsize uwtable "target-cpu"="x86-64" }
