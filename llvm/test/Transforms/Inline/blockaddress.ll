; RUN: opt -inline -S < %s | FileCheck %s
; PR10162

; Make sure the blockaddress is mapped correctly when doit is inlined
; CHECK: store i8* blockaddress(@f, %here.i), i8** @ptr1, align 8

@i = global i32 1, align 4
@ptr1 = common global i8* null, align 8

define void @doit(i8** nocapture %pptr, i32 %cond) nounwind uwtable {
entry:
  %tobool = icmp eq i32 %cond, 0
  br i1 %tobool, label %if.end, label %here

here:
  store i8* blockaddress(@doit, %here), i8** %pptr, align 8
  br label %if.end

if.end:
  ret void
}

define void @f(i32 %cond) nounwind uwtable {
entry:
  call void @doit(i8** @ptr1, i32 %cond)
  ret void
}
