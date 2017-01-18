; RUN: opt < %s -inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s
; Do not inline calls with variable-sized alloca.

@q = common global i8* null

define i8* @a(i32 %i) nounwind {
; CHECK-LABEL: define i8* @a
entry:
  %i_addr = alloca i32
  %retval = alloca i8*
  %p = alloca i8*
  %"alloca point" = bitcast i32 0 to i32
  store i32 %i, i32* %i_addr
  %0 = load i32, i32* %i_addr, align 4
  %1 = alloca i8, i32 %0
  store i8* %1, i8** %p, align 4
  %2 = load i8*, i8** %p, align 4
  store i8* %2, i8** @q, align 4
  br label %return

return:
  %retval1 = load i8*, i8** %retval
  ret i8* %retval1
}

define void @b(i32 %i) nounwind {
; CHECK-LABEL: define void @b
entry:
  %i_addr = alloca i32
  %"alloca point" = bitcast i32 0 to i32
  store i32 %i, i32* %i_addr
  %0 = load i32, i32* %i_addr, align 4
  %1 = call i8* @a(i32 %0) nounwind
; CHECK: call i8* @a
  br label %return

return:
  ret void
}
