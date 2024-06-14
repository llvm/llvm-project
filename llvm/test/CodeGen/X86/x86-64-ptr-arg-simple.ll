; RUN: llc -mtriple=x86_64-pc-linux < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-pc-linux-gnux32 < %s | FileCheck -check-prefix=X32ABI %s

; %in is kept in %esi for both ABIs. But the pointer will be passed in %edi
; for x32, not %rdi

; CHECK: movl %esi, (%rdi)
; X32ABI: movl %esi, (%edi)

define void @foo(ptr nocapture %out, i32 %in) nounwind {
entry:
  store i32 %in, ptr %out, align 4
  ret void
}

; CHECK: bar
; CHECK: movl (%rsi), %eax

; Similarly here, but for loading
; X32ABI: bar
; X32ABI: movl (%esi), %eax

define void @bar(ptr nocapture %pOut, ptr nocapture %pIn) nounwind {
entry:
  %0 = load i32, ptr %pIn, align 4
  store i32 %0, ptr %pOut, align 4
  ret void
}

