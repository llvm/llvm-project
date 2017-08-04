; RUN: opt -inline < %s

define void @patatino() {
for.cond:
  br label %for.body

for.body:
  %tobool = icmp eq i32 5, 0
  %sel = select i1 %tobool, i32 0, i32 2
  br i1 undef, label %cleanup1.thread, label %cleanup1

cleanup1.thread:
  ret void

cleanup1:
  %cleanup.dest2 = phi i32 [ %sel, %for.body ]
  %switch = icmp ult i32 %cleanup.dest2, 1
  ret void
}

define void @main() {
entry:
  call void @patatino()
  ret void
}
