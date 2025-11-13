; RUN: llc < %s -mtriple=x86_64 | FileCheck %s

define void @f(ptr %0, i64 %1) {
BB:
  %fps = load <2 x ptr>, ptr %0
  %fp = extractelement <2 x ptr> %fps, i64 %1
  %p = call ptr %fp(i32 42)
  store <2 x ptr> %fps, ptr %p
  ret void
}
