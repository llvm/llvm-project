; Bugzilla: https://bugs.llvm.org/show_bug.cgi?id=33623
; RUN: llvm-diff %s %s

%struct.it = type { i64, ptr }

@a_vector = internal global [2 x i64] zeroinitializer, align 16

define i32 @foo(ptr %it) {

entry:
  %a = getelementptr inbounds %struct.it, ptr %it, i64 0, i32 1
  store <2 x ptr> <ptr @a_vector, ptr @a_vector>, ptr %a, align 8

  ret i32 0
}
