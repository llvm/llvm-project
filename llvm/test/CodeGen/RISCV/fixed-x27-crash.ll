; RUN: llc < %s -mtriple=riscv64-unknown-fuchsia -mattr=+reserve-x27 -verify-machineinstrs

define i64 @foo() {
entry:
  %local = alloca i64
  br label %end

end:                                           ; preds = %entry
  store i64 0, ptr %local, align 8
  %0 = tail call i64 @llvm.read_register.i64(metadata !0)
  ret i64 %0
}

declare i64 @llvm.read_register.i64(metadata)

!0 = !{!"s11"}
