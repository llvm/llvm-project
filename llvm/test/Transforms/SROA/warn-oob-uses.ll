; RUN: opt -passes=sroa -disable-output %s 2>&1 | FileCheck %s

define i8 @load_starts_outside_alloca() {
entry:
  %alloca = alloca i32, align 4
  %oob = getelementptr i8, ptr %alloca, i64 4
  %load = load i8, ptr %oob, align 1
  ret i8 %load
}

; CHECK: warning: {{.*}}Potential OOB use: 1 bytes, offset 4

define void @store_extends_past_alloca() {
entry:
  %alloca = alloca i8, align 1
  store i16 0, ptr %alloca, align 1
  ret void
}

; CHECK: warning: {{.*}}Potential OOB use: 2 bytes, offset 0
