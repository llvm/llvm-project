; RUN: opt -passes=gvn -S %s | FileCheck %s

%MyStruct = type { i32, i32 }
define i8 @foo(i64 %in, ptr %arr, i1 %arg) {
  %addr = alloca %MyStruct
  %dead = trunc i64 %in to i32
  br i1 %arg, label %next, label %tmp

tmp:
  call void @bar()
  br label %next

next:
  store i64 %in, ptr %addr
  br label %final

final:
  %idx32 = load i32, ptr %addr

; CHECK: %resptr = getelementptr i8, ptr %arr, i32 %dead
  %resptr = getelementptr i8, ptr %arr, i32 %idx32
  %res = load i8, ptr %resptr

  ret i8 %res
}

declare void @bar()
