; RUN: opt -passes=aa-eval -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck %s

; CHECK: NoAlias: i8* %a, i8* %gep
define void @inttoptr_alloca() {
  %a = alloca i8
  %a.int = ptrtoint ptr %a to i64
  %a.int.1 = add i64 %a.int, 1
  %gep = getelementptr i8, ptr inttoptr (i64 -1 to ptr), i64 %a.int.1
  %load = load i8, ptr %gep
  store i8 1, ptr %a
  ret void
}

; CHECK: NoAlias: i8* %a, i8* %gep
define void @inttoptr_alloca_unknown_relation(i64 %offset) {
  %a = alloca i8
  %a.int = ptrtoint ptr %a to i64
  %gep = getelementptr i8, ptr inttoptr (i64 -1 to ptr), i64 %offset
  %load = load i8, ptr %gep
  store i8 1, ptr %a
  ret void
}

; CHECK: NoAlias: i8* %a, i8* %gep
define void @inttoptr_alloca_noescape(i64 %offset) {
  %a = alloca i8
  %gep = getelementptr i8, ptr inttoptr (i64 -1 to ptr), i64 %offset
  %load = load i8, ptr %gep
  store i8 1, ptr %a
  ret void
}

; CHECK: NoAlias: i8* %a, i8* %gep
define void @inttoptr_noalias(ptr noalias %a) {
  %a.int = ptrtoint ptr %a to i64
  %a.int.1 = add i64 %a.int, 1
  %gep = getelementptr i8, ptr inttoptr (i64 -1 to ptr), i64 %a.int.1
  %load = load i8, ptr %gep
  store i8 1, ptr %a
  ret void
}

; CHECK: NoAlias: i8* %a, i8* %gep
define void @inttoptr_noalias_noescape(ptr noalias %a, i64 %offset) {
  %gep = getelementptr i8, ptr inttoptr (i64 -1 to ptr), i64 %offset
  %load = load i8, ptr %gep
  store i8 1, ptr %a
  ret void
}
