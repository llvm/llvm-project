; RUN: llc < %s -O0 -mtriple=x86_64-unknown-linux-gnu
; https://bugs.llvm.org/show_bug.cgi?id=51699

%"[]u8" = type { i8*, i64 }
%std.mem.Allocator = type { void ({ %"[]u8", i16 }*, %std.builtin.StackTrace*, %std.mem.Allocator*, i64, i29, i29, i64)*, void ({ i64, i16 }*, %std.builtin.StackTrace*, %std.mem.Allocator*, %"[]u8"*, i29, i64, i29, i64)* }
%std.builtin.StackTrace = type { i64, %"[]usize" }
%"[]usize" = type { i64*, i64 }

define  void @fun(%"[]u8"* %0) #0 {
Entry:
  %1 = alloca [6 x i64], align 8
  br label %ErrRetContinue

ErrRetContinue:                                   ; preds = %Entry
  %2 = call i64 asm sideeffect "rolq $$3,  %rdi ; rolq $$13, %rdi\0Arolq $$61, %rdi ; rolq $$51, %rdi\0Axchgq %rbx,%rbx\0A", "={rdx},{rax},0,~{cc},~{memory}"(i64 undef, i64 0)
  %3 = call fastcc i64 undef(%std.mem.Allocator* undef, %"[]u8"* %0, i29 undef, i64 0, i29 0, i64 undef)
  ret void
}

attributes #0 = { sspstrong }
