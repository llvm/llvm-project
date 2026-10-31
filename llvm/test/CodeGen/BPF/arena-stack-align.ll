; RUN: opt -S -mtriple=bpfel -passes=bpf-arena-stack < %s | FileCheck %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+arena-stack < %s | FileCheck --check-prefix=ASM %s
;
; r12 is aligned to 16 bytes. A local that wants more gets room for that
; and its address is aligned when the function runs.

define internal void @use(ptr %p) noinline {
  store i64 1, ptr %p
  ret void
}

define i64 @f() {
; CHECK-LABEL: @f(
; CHECK:       %a = alloca [176 x i8], align 16, !bpf.arena.stack
; CHECK-NEXT:  [[END:%.*]] = getelementptr i8, ptr %a, i64 112
; CHECK-NEXT:  [[P:%.*]] = call ptr @llvm.ptrmask.p0.i64(ptr [[END]], i64 -128)
; CHECK-NOT:   lifetime
; CHECK:       call void @use(ptr [[P]])
; ASM-LABEL:   f:
; ASM:         r12 += 176
; ASM:         r[[R:[0-9]]] = r12
; ASM-NEXT:    r[[R]] += -64
; ASM-NEXT:    r[[R]] &= -128
; ASM:         call use
  %a = alloca [64 x i8], align 128
  call void @llvm.lifetime.start.p0(ptr %a)
  call void @use(ptr %a)
  %v = load i64, ptr %a
  call void @llvm.lifetime.end.p0(ptr %a)
  ret i64 %v
}

define void @plain() {
; CHECK-LABEL: @plain(
; CHECK:       %a = alloca [64 x i8], align 16, !bpf.arena.stack
; CHECK-NEXT:  call void @use(ptr %a)
  %a = alloca [64 x i8], align 16
  call void @use(ptr %a)
  ret void
}
