; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic -filetype=obj -o /dev/null
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic | FileCheck %s

@a = dso_local global i16 0, align 2
@b = dso_local global [1 x i8] zeroinitializer, align 1
@c = dso_local global i64 0, align 8
@d = dso_local global ptr null, align 8
@e = dso_local global i32 0, align 4
@f = dso_local global i32 0, align 4

define dso_local void @test_rip_relative_with_index() {
; CHECK-LABEL: test_rip_relative_with_index:
; CHECK:       # %bb.0:
; CHECK-NOT:     (%{{[a-z0-9]+}},%rip)
; CHECK-NOT:     (%rip,%{{[a-z0-9]+}})
entry:
  %0 = load ptr, ptr @d, align 8
  %1 = load i32, ptr %0, align 4
  %2 = load i32, ptr @f, align 4
  %and = and i32 %2, 5
  %rem = and i32 %1, 1
  %add = add nuw nsw i32 %and, %rem
  store i32 %add, ptr @e, align 4
  %c.promoted = load i64, ptr @c, align 8
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %h.014 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %xor1113 = phi i64 [ %c.promoted, %entry ], [ %xor, %for.body ]
  %add1 = add nuw nsw i32 %h.014, %add
  %rem2 = urem i32 %add1, 5
  %idxprom = zext nneg i32 %rem2 to i64
  %arrayidx = getelementptr inbounds nuw [4 x i8], ptr %0, i64 %idxprom
  %3 = load i32, ptr %arrayidx, align 4
  %conv = sext i32 %3 to i64
  %xor = xor i64 %xor1113, %conv
  %inc = add nuw nsw i32 %h.014, 1
  %exitcond.not = icmp eq i32 %inc, 5
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx6.le = getelementptr inbounds nuw i8, ptr @b, i64 %idxprom
  %4 = load i8, ptr %arrayidx6.le, align 1
  %conv7.le = sext i8 %4 to i16
  store i64 %xor, ptr @c, align 8
  store i16 %conv7.le, ptr @a, align 2
  ret void
}
