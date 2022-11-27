; RUN: opt < %s -jump-threading -disable-output

%struct.aaa = type { i8 }

define void @chrome(ptr noalias sret(%struct.aaa) %arg) local_unnamed_addr #0 align 2 personality ptr @chrome2 {
bb:
  %tmp = load i32, ptr undef, align 4
  %tmp1 = icmp eq i32 %tmp, 0
  br i1 %tmp1, label %bb2, label %bb13

bb2:
  %tmp4 = load i8, ptr %arg, align 1
  %tmp5 = icmp eq i8 %tmp4, 0
  br i1 %tmp5, label %bb6, label %bb7

bb6:
  store i8 0, ptr %arg, align 1
  br label %bb7

bb7:
  %tmp8 = load i8, ptr %arg, align 1
  %tmp9 = icmp ne i8 %tmp8, 0
  %tmp10 = select i1 %tmp9, i1 true, i1 false
  br i1 %tmp10, label %bb12, label %bb11

bb11:
  br label %bb12

bb12:
  br i1 %tmp9, label %bb14, label %bb13

bb13:
  unreachable

bb14:
  ret void
}

declare i32 @chrome2(...)
