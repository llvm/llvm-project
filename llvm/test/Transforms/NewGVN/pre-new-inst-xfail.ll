; XFAIL: *
; RUN: opt -basic-aa -newgvn -S %s | FileCheck %s

%MyStruct = type { i32, i32 }
define i8 @foo(i64 %in, i8* %arr) {
  %addr = alloca %MyStruct
  %dead = trunc i64 %in to i32
  br i1 undef, label %next, label %tmp

tmp:
  call void @bar()
  br label %next

next:
  %addr64 = bitcast %MyStruct* %addr to i64*
  store i64 %in, i64* %addr64
  br label %final

final:
  %addr32 = getelementptr %MyStruct, %MyStruct* %addr, i32 0, i32 0
  %idx32 = load i32, i32* %addr32

; CHECK: %resptr = getelementptr i8, i8* %arr, i32 %dead
  %resptr = getelementptr i8, i8* %arr, i32 %idx32
  %res = load i8, i8* %resptr

  ret i8 %res
}

declare void @bar()
