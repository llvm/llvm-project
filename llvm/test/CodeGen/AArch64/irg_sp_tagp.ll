; RUN: llc < %s -mtriple=aarch64 -mattr=+mte | FileCheck %s

define dso_local i8* @small_alloca() {
entry:
; CHECK-LABEL: small_alloca:
; CHECK:      irg  x0, sp{{$}}
; CHECK:      ret
  %a = alloca i8, align 16
  %q = call i8* @llvm.aarch64.irg.sp(i64 0)
  %q1 = call i8* @llvm.aarch64.tagp.p0i8(i8* %a, i8* %q, i64 1)
  ret i8* %q1
}

@sink = dso_local global i8* null, align 8

; Check that IRG is pinned to %b because the store instruction needs
; the address in a non-fixed physical register and can benefit from it
; being equal to the base tagged pointer.
define dso_local i8* @small_allocas() {
entry:
; CHECK-LABEL: small_allocas:
; CHECK:      irg  [[R:x[0-9]+]], sp{{$}}
; CHECK:      addg x0, [[R]], #16, #1
; CHECK:      str  [[R]], {{.*}}sink
; CHECK:      ret
  %a = alloca i8, align 16
  %b = alloca i8, align 16
  %q = call i8* @llvm.aarch64.irg.sp(i64 0)
  %q1 = call i8* @llvm.aarch64.tagp.p0i8(i8* %a, i8* %q, i64 1)
  %q2 = call i8* @llvm.aarch64.tagp.p0i8(i8* %b, i8* %q, i64 2)
  store i8* %q2, i8** @sink, align 8
  ret i8* %q1
}

; Two large allocas. One's offset overflows addg immediate.
define dso_local void @huge_allocas() {
entry:
; CHECK-LABEL: huge_allocas:
; CHECK:      irg  x1, sp{{$}}
; CHECK:      add  [[TMP:x[0-9]+]], x1, #3088
; CHECK:      addg x0, [[TMP]], #1008, #1
; CHECK:      bl use2
  %a = alloca i8, i64 4096, align 16
  %b = alloca i8, i64 4096, align 16
  %base = call i8* @llvm.aarch64.irg.sp(i64 0)
  %a_t = call i8* @llvm.aarch64.tagp.p0i8(i8* %a, i8* %base, i64 1)
  %b_t = call i8* @llvm.aarch64.tagp.p0i8(i8* %b, i8* %base, i64 0)
  call void @use2(i8* %a_t, i8* %b_t)
  ret void
}

; Realigned stack frame. IRG uses value of SP after realignment,
; ADDG for the first stack allocation has offset 0.
define dso_local void @realign() {
entry:
; CHECK-LABEL: realign:
; CHECK:      mov  x29, sp
; CHECK:      and  sp, x{{[0-9]*}}, #0xffffffffffffffc0
; CHECK:      irg  x0, sp{{$}}
; CHECK:      bl use
  %a = alloca i8, i64 4096, align 64
  %base = call i8* @llvm.aarch64.irg.sp(i64 0)
  %a_t = call i8* @llvm.aarch64.tagp.p0i8(i8* %a, i8* %base, i64 1)
  call void @use(i8* %a_t)
  ret void
}

; With a dynamic alloca, IRG has to use FP with non-zero offset.
; ADDG offset for the single static alloca is still zero.
define dso_local void @dynamic_alloca(i64 %size) {
entry:
; CHECK-LABEL: dynamic_alloca:
; CHECK:      sub  x1, x29, #[[OFS:[0-9]+]]
; CHECK:      irg  x1, x1
; CHECK-DAG:  sub  x0, x29, #[[OFS]]
; CHECK:      bl   use2
  %base = call i8* @llvm.aarch64.irg.sp(i64 0)
  %a = alloca i128, i64 %size, align 16
  %b = alloca i8, i64 16, align 16
  %b_t = call i8* @llvm.aarch64.tagp.p0i8(i8* %b, i8* %base, i64 1)
  call void @use2(i8* %b, i8* %b_t)
  ret void
}

; Both dynamic alloca and realigned frame.
; After initial realignment, generate the base pointer.
; IRG uses the base pointer w/o offset.
; Offsets for tagged and untagged pointers to the same alloca match.
define dso_local void @dynamic_alloca_and_realign(i64 %size) {
entryz:
; CHECK-LABEL: dynamic_alloca_and_realign:
; CHECK:      and  sp, x{{.*}}, #0xffffffffffffffc0
; CHECK:      mov  x19, sp
; CHECK:      add  x1, x19, #[[OFS:[0-9]+]]
; CHECK:      irg  x1, x1
; CHECK-DAG:  add  x0, x19, #[[OFS]]
; CHECK:      bl   use2
  %base = call i8* @llvm.aarch64.irg.sp(i64 0)
  %a = alloca i128, i64 %size, align 64
  %b = alloca i8, i64 16, align 16
  %b_t = call i8* @llvm.aarch64.tagp.p0i8(i8* %b, i8* %base, i64 1)
  call void @use2(i8* %b, i8* %b_t)
  ret void
}

declare void @use(i8*)
declare void @use2(i8*, i8*)

declare i8* @llvm.aarch64.irg.sp(i64 %exclude)
declare i8* @llvm.aarch64.tagp.p0i8(i8* %p, i8* %tag, i64 %ofs)
