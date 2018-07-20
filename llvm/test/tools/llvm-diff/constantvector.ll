; Bugzilla: https://bugs.llvm.org/show_bug.cgi?id=33623
; RUN: llvm-diff %s %s

%struct.it = type { i64, i64* }

@a_vector = internal global [2 x i64] zeroinitializer, align 16

define i32 @foo(%struct.it* %it) {

entry:
  %a = getelementptr inbounds %struct.it, %struct.it* %it, i64 0, i32 1
  %tmp0 = bitcast i64** %a to <2 x i64*>*
  store <2 x i64*> <i64* getelementptr inbounds ([2 x i64], [2 x i64]* @a_vector, i64 0, i64 0), i64* getelementptr inbounds ([2 x i64], [2 x i64]* @a_vector, i64 0, i64 0)>, <2 x i64*>* %tmp0, align 8

  ret i32 0
}
