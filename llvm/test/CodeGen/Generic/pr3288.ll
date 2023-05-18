; RUN: llc < %s
; PR3288

define void @a() {
  %i = insertvalue [2 x [2 x i32]] undef, [2 x i32] undef, 1
  ret void
}
define void @b() {
  %i = insertvalue {{i32,float},{i16,double}} undef, {i16,double} undef, 1
  ret void
}
define void @c() {
  %i = insertvalue [2 x [2 x i32]] zeroinitializer, [2 x i32] zeroinitializer, 1
  ret void
}
define void @d() {
  %i = insertvalue {{i32,float},{i16,double}} zeroinitializer, {i16,double} zeroinitializer, 1
  ret void
}
define void @e() {
  %i = insertvalue [2 x [2 x i32]] undef, [2 x i32] undef, 0
  ret void
}
define void @f() {
  %i = insertvalue {{i32,float},{i16,double}} undef, {i32,float} undef, 0
  ret void
}
define void @g() {
  %i = insertvalue [2 x [2 x i32]] zeroinitializer, [2 x i32] zeroinitializer, 0
  ret void
}
define void @h() {
  %i = insertvalue {{i32,float},{i16,double}} zeroinitializer, {i32,float} zeroinitializer, 0
  ret void
}
define void @ax() {
  %i = insertvalue [2 x [2 x i32]] undef, i32 undef, 1, 1
  ret void
}
define void @bx() {
  %i = insertvalue {{i32,float},{i16,double}} undef, double undef, 1, 1
  ret void
}
define void @cx() {
  %i = insertvalue [2 x [2 x i32]] zeroinitializer, i32 zeroinitializer, 1, 1
  ret void
}
define void @dx() {
  %i = insertvalue {{i32,float},{i16,double}} zeroinitializer, double zeroinitializer, 1, 1
  ret void
}
define void @ex() {
  %i = insertvalue [2 x [2 x i32]] undef, i32 undef, 0, 1
  ret void
}
define void @fx() {
  %i = insertvalue {{i32,float},{i16,double}} undef, float undef, 0, 1
  ret void
}
define void @gx() {
  %i = insertvalue [2 x [2 x i32]] zeroinitializer, i32 zeroinitializer, 0, 1
  ret void
}
define void @hx() {
  %i = insertvalue {{i32,float},{i16,double}} zeroinitializer, float zeroinitializer, 0, 1
  ret void
}
