; RUN: opt -passes=globalopt -S < %s | FileCheck %s

@a = internal global i32 0, align 4
@b = internal global i32 0, align 4
@c = internal global i32 0, align 4
@d = internal constant [4 x i8] c"foo\00", align 1
@e = linkonce_odr global i32 0
@f = internal addrspace(3) global float undef, align 4

; CHECK: @a = internal global i32 0, align 4
; CHECK: @b = internal global i32 0, align 4
; CHECK: @c = internal unnamed_addr global i32 0, align 4
; CHECK: @d = internal unnamed_addr constant [4 x i8] c"foo\00", align 1
; CHECK: @e = linkonce_odr local_unnamed_addr global i32 0
; CHECK: @f = internal unnamed_addr addrspace(3) global float undef, align 4

; CHECK: define internal fastcc void @used_internal() unnamed_addr {
define internal void @used_internal() {
  ret void
}

define i32 @get_e() {
       call void @used_internal()
       %t = load i32, ptr @e
       ret i32 %t
}

define void @set_e(i32 %x) {
       store i32 %x, ptr @e
       ret void
}

define i1 @bah(i64 %i) nounwind readonly optsize ssp {
entry:
  %arrayidx4 = getelementptr inbounds [4 x i8], ptr @d, i64 0, i64 %i
  %tmp5 = load i8, ptr %arrayidx4, align 1
  %tmp6 = load i8, ptr @d, align 1
  %cmp = icmp eq i8 %tmp5, %tmp6
  ret i1 %cmp
}

define void @baz(i32 %x) {
entry:
  store i32 %x, ptr @a, align 4
  store i32 %x, ptr @b, align 4
  store i32 %x, ptr @c, align 4
  ret void
}

define i32 @foo(ptr %x) nounwind readnone optsize ssp {
entry:
  %cmp = icmp eq ptr %x, @a
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define i32 @bar() {
entry:
  switch i64 ptrtoint (ptr @b to i64), label %sw.epilog [
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
  %tmp1 = load i32, ptr @c, align 4
  ret i32 %tmp1
}

define float @use_addrspace_cast_for_load() {
  %p = addrspacecast ptr addrspace(3) @f to ptr
  %v = load float, ptr %p
  ret float %v
}

define void @use_addrspace_cast_for_store(float %x) {
  %p = addrspacecast ptr addrspace(3) @f to ptr
  store float %x, ptr %p
  ret void
}
