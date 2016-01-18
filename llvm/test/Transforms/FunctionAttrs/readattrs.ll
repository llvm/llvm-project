; RUN: opt < %s -functionattrs -S | FileCheck %s
@x = global i32 0

declare void @test1_1(i8* %x1_1, i8* readonly %y1_1, ...)

; CHECK: define void @test1_2(i8* %x1_2, i8* readonly %y1_2, i8* %z1_2)
define void @test1_2(i8* %x1_2, i8* %y1_2, i8* %z1_2) {
  call void (i8*, i8*, ...) @test1_1(i8* %x1_2, i8* %y1_2, i8* %z1_2)
  store i32 0, i32* @x
  ret void
}

; CHECK: define i8* @test2(i8* readnone %p)
define i8* @test2(i8* %p) {
  store i32 0, i32* @x
  ret i8* %p
}

; CHECK: define i1 @test3(i8* readnone %p, i8* readnone %q)
define i1 @test3(i8* %p, i8* %q) {
  %A = icmp ult i8* %p, %q
  ret i1 %A
}

declare void @test4_1(i8* nocapture) readonly

; CHECK: define void @test4_2(i8* nocapture readonly %p)
define void @test4_2(i8* %p) {
  call void @test4_1(i8* %p)
  ret void
}

; CHECK: define void @test5(i8** nocapture %p, i8* %q)
; Missed optz'n: we could make %q readnone, but don't break test6!
define void @test5(i8** %p, i8* %q) {
  store i8* %q, i8** %p
  ret void
}

declare void @test6_1()
; CHECK: define void @test6_2(i8** nocapture %p, i8* %q)
; This is not a missed optz'n.
define void @test6_2(i8** %p, i8* %q) {
  store i8* %q, i8** %p
  call void @test6_1()
  ret void
}

; CHECK: define void @test7_1(i32* inalloca nocapture %a)
; inalloca parameters are always considered written
define void @test7_1(i32* inalloca %a) {
  ret void
}

; CHECK: define i32* @test8_1(i32* readnone %p)
define i32* @test8_1(i32* %p) {
entry:
  ret i32* %p
}

; CHECK: define void @test8_2(i32* %p)
define void @test8_2(i32* %p) {
entry:
  %call = call i32* @test8_1(i32* %p)
  store i32 10, i32* %call, align 4
  ret void
}
