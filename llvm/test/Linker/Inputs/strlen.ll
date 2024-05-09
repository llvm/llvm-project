target triple = "x86_64-unknown-linux-gnu"

define i64 @strlen(ptr %s) #0 {
entry:
  br label %for.cond

for.cond:
  %s.addr.0 = phi ptr [ %s, %entry ], [ %incdec.ptr, %for.cond ]
  %0 = load i8, ptr %s.addr.0, align 1
  %tobool.not = icmp eq i8 %0, 0
  %incdec.ptr = getelementptr inbounds i8, ptr %s.addr.0, i64 1
  br i1 %tobool.not, label %for.end, label %for.cond

for.end:
  %sub.ptr.lhs.cast = ptrtoint ptr %s.addr.0 to i64
  %sub.ptr.rhs.cast = ptrtoint ptr %s to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  ret i64 %sub.ptr.sub
}

attributes #0 = { noinline }
