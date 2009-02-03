; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | grep unreachable
; PR1796

declare void @Finisher(i32) noreturn

define void @YYY(i32) {
  tail call void @Finisher(i32 %0) noreturn
  tail call void @Finisher(i32 %0) noreturn
  ret void
}

