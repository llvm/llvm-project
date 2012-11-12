; RUN: opt %s -globalopt -S | FileCheck %s

@a = internal global i32 0, align 4
@b = internal global i32 0, align 4
@c = internal global i32 0, align 4
@d = internal constant [4 x i8] c"foo\00", align 1
@e = linkonce_odr global i32 0

; CHECK: @a = internal global i32 0, align 4
; CHECK: @b = internal global i32 0, align 4
; CHECK: @c = internal unnamed_addr global i32 0, align 4
; CHECK: @d = internal unnamed_addr constant [4 x i8] c"foo\00", align 1
; CHECK: @e = linkonce_odr global i32 0

define i32 @get_e() {
       %t = load i32* @e
       ret i32 %t
}

define void @set_e(i32 %x) {
       store i32 %x, i32* @e
       ret void
}

define i1 @bah(i64 %i) nounwind readonly optsize ssp {
entry:
  %arrayidx4 = getelementptr inbounds [4 x i8]* @d, i64 0, i64 %i
  %tmp5 = load i8* %arrayidx4, align 1
  %array0 = bitcast [4 x i8]* @d to i8*
  %tmp6 = load i8* %array0, align 1
  %cmp = icmp eq i8 %tmp5, %tmp6
  ret i1 %cmp
}

define void @baz(i32 %x) {
entry:
  store i32 %x, i32* @a, align 4
  store i32 %x, i32* @b, align 4
  store i32 %x, i32* @c, align 4
  ret void
}

define i32 @foo(i32* %x) nounwind readnone optsize ssp {
entry:
  %cmp = icmp eq i32* %x, @a
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @bar() {
entry:
  switch i64 ptrtoint (i32* @b to i64), label %sw.epilog [
    i64 1, label %return
    i64 0, label %return
  ]

sw.epilog:
  ret i32 0

return:
  ret i32 1
}

define i32 @zed() {
entry:
  %tmp1 = load i32* @c, align 4
  ret i32 %tmp1
}
