; RUN: llvm-as < %s | llvm-dis > %t
; RUN: grep insertvalue %t | count 1
; RUN: grep extractvalue %t | count 1

define float @foo({{i32},{float, double}}* %p) nounwind {
  %t = load {{i32},{float, double}}* %p
  %s = extractvalue {{i32},{float, double}} %t, 1, 0
  %r = insertvalue {{i32},{float, double}} %t, double 2.0, 1, 1
  store {{i32},{float, double}} %r, {{i32},{float, double}}* %p
  ret float %s
}
define float @bar({{i32},{float, double}}* %p) nounwind {
  store {{i32},{float, double}} insertvalue ({{i32},{float, double}}{{i32}{i32 4},{float, double}{float 4.0, double 5.0}}, double 20.0, 1, 1), {{i32},{float, double}}* %p
  ret float extractvalue ({{i32},{float, double}}{{i32}{i32 3},{float, double}{float 7.0, double 9.0}}, 1, 0)
}
define float @car({{i32},{float, double}}* %p) nounwind {
  store {{i32},{float, double}} insertvalue ({{i32},{float, double}} undef, double 20.0, 1, 1), {{i32},{float, double}}* %p
  ret float extractvalue ({{i32},{float, double}} undef, 1, 0)
}
define float @dar({{i32},{float, double}}* %p) nounwind {
  store {{i32},{float, double}} insertvalue ({{i32},{float, double}} zeroinitializer, double 20.0, 1, 1), {{i32},{float, double}}* %p
  ret float extractvalue ({{i32},{float, double}} zeroinitializer, 1, 0)
}
