; RUN: opt -inline -S < %s | FileCheck %s
; PR10162

; Make sure doit is not inlined since the blockaddress is taken
; which could be unsafe
; CHECK: store i8* blockaddress(@doit, %here), i8** %pptr, align 8

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

; PR27233: We can inline @run into @init.  Don't crash on it.
;
; CHECK-LABEL: define void @init
; CHECK:         store i8* blockaddress(@run, %bb)
; CHECK-SAME:        @run.bb
define void @init() {
entry:
  call void @run()
  ret void
}

define void @run() {
entry:
  store i8* blockaddress(@run, %bb), i8** getelementptr inbounds ([1 x i8*], [1 x i8*]* @run.bb, i64 0, i64 0), align 8
  ret void

bb:
  unreachable
}

@run.bb = global [1 x i8*] zeroinitializer
