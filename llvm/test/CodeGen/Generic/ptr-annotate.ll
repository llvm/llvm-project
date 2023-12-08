; RUN: llc < %s

; PR15253

%struct.mystruct = type { i32 }

@.str = private unnamed_addr constant [4 x i8] c"sth\00", section "llvm.metadata"
@.str1 = private unnamed_addr constant [4 x i8] c"t.c\00", section "llvm.metadata"

define void @foo() {
entry:
  %m = alloca i8, align 4
  %0 = call ptr @llvm.ptr.annotation.p0(ptr %m, ptr @.str, ptr @.str1, i32 2, ptr null)
  store i8 1, ptr %0, align 4
  ret void
}

declare ptr @llvm.ptr.annotation.p0(ptr, ptr, ptr, i32, ptr) #1
