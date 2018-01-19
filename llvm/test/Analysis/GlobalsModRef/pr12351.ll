; RUN: opt < %s -basicaa -globals-aa -gvn -S -disable-verify | FileCheck %s

declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i1)
define void @foo(i8* %x, i8* %y) {
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %x, i8* %y, i32 1, i1 false);
  ret void
}

define void @bar(i8* %y, i8* %z) {
  %x = alloca i8
  call void @foo(i8* %x, i8* %y)
  %t = load i8, i8* %x
  store i8 %t, i8* %y
; CHECK: store i8 %t, i8* %y
  ret void
}


define i32 @foo2() {
  %foo = alloca i32
  call void @bar2(i32* %foo)
  %t0 = load i32, i32* %foo, align 4
; CHECK: %t0 = load i32, i32* %foo, align 4
  ret i32 %t0
}

define void @bar2(i32* %foo)  {
  store i32 0, i32* %foo, align 4
  tail call void @llvm.dbg.value(metadata !{}, i64 0, metadata !{}, metadata !{})
  ret void
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone
