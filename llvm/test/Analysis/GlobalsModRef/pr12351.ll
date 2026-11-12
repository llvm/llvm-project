; RUN: opt < %s -aa-pipeline=basic-aa,globals-aa -passes=gvn -S -disable-verify | FileCheck %s

declare void @llvm.memcpy.p0.p0.i32(ptr, ptr, i32, i1)
define void @foo(ptr %x, ptr %y) {
  call void @llvm.memcpy.p0.p0.i32(ptr %x, ptr %y, i32 1, i1 false);
  ret void
}

define void @bar(ptr %y, ptr %z) {
  %x = alloca i8
  call void @foo(ptr %x, ptr %y)
  %t = load i8, ptr %x
  store i8 %t, ptr %y
; CHECK: store i8 %t, ptr %y
  ret void
}


define i32 @foo2() {
  %foo = alloca i32
  call void @bar2(ptr %foo)
  %t0 = load i32, ptr %foo, align 4
; CHECK: %t0 = load i32, ptr %foo, align 4
  ret i32 %t0
}

define void @bar2(ptr %foo)  {
  store i32 0, ptr %foo, align 4
  tail call void @llvm.dbg.value(metadata !{}, i64 0, metadata !{}, metadata !{})
  ret void
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone
