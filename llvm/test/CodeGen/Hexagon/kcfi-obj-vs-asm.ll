;; The compiler and the assembler must agree on the encoding of a KCFI check:
;; LowerKCFI_CHECK() emitting packets by hand makes the two paths independent
;; implementations of the same sequence.  Comparing the .text images needs no
;; prediction of the right encoding, so it catches divergences nobody thought
;; to write a CHECK line for.
;;
;; Debugging a failure: this only says the two paths disagree somewhere in the
;; module, and they are not identically configured -- the asm parser
;; canonicalizes with a checker and AttemptCompatibility, the printer with
;; neither -- so a packet only the perf checker objects to can legitimately
;; differ.  Check kcfi-packetization.ll first, then bisect by deleting
;; functions.  It also covers .kcfi_traps, which objcopy cannot extract here
;; because its relocations point at .text.

; RUN: llc -mtriple=hexagon -filetype=obj < %s -o %t.direct.o
; RUN: llc -mtriple=hexagon -filetype=asm < %s -o %t.s
; RUN: llvm-mc -triple=hexagon -filetype=obj %t.s -o %t.viaasm.o
; RUN: llvm-objcopy -O binary --only-section=.text %t.direct.o %t.direct.bin
; RUN: llvm-objcopy -O binary --only-section=.text %t.viaasm.o %t.viaasm.bin
; RUN: cmp %t.direct.bin %t.viaasm.bin

; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -filetype=obj < %s -o %t.68.o
; RUN: llc -mtriple=hexagon -mcpu=hexagonv68 -filetype=asm < %s -o %t.68.s
; RUN: llvm-mc -triple=hexagon -mcpu=hexagonv68 -filetype=obj %t.68.s -o %t.68a.o
; RUN: llvm-objcopy -O binary --only-section=.text %t.68.o %t.68.bin
; RUN: llvm-objcopy -O binary --only-section=.text %t.68a.o %t.68a.bin
; RUN: cmp %t.68.bin %t.68a.bin

; RUN: llc -mtriple=hexagon -mcpu=hexagonv79 -filetype=obj < %s -o %t.79.o
; RUN: llc -mtriple=hexagon -mcpu=hexagonv79 -filetype=asm < %s -o %t.79.s
; RUN: llvm-mc -triple=hexagon -mcpu=hexagonv79 -filetype=obj %t.79.s -o %t.79a.o
; RUN: llvm-objcopy -O binary --only-section=.text %t.79.o %t.79.bin
; RUN: llvm-objcopy -O binary --only-section=.text %t.79a.o %t.79a.bin
; RUN: cmp %t.79.bin %t.79a.bin

;; The configuration the Hexagon Linux kernel actually builds with.
; RUN: llc -mtriple=hexagon --disable-packetizer -filetype=obj < %s -o %t.np.o
; RUN: llc -mtriple=hexagon --disable-packetizer -filetype=asm < %s -o %t.np.s
; RUN: llvm-mc -triple=hexagon -filetype=obj %t.np.s -o %t.npa.o
; RUN: llvm-objcopy -O binary --only-section=.text %t.np.o %t.np.bin
; RUN: llvm-objcopy -O binary --only-section=.text %t.npa.o %t.npa.bin
; RUN: cmp %t.np.bin %t.npa.bin

;; Hash needing an extender, target in r0.
define void @plain(ptr noundef %fp) {
  call void %fp() [ "kcfi"(i32 12345678) ]
  ret void
}

;; Hash small enough that a naive implementation might skip the extender.
define void @small(ptr noundef %fp) {
  call void %fp() [ "kcfi"(i32 7) ]
  ret void
}

;; Hash with the top bit set: sign-extension mistakes show up here.
define void @negative_hash(ptr noundef %fp) {
  call void %fp() [ "kcfi"(i32 -559038737) ]
  ret void
}

;; Six integer arguments occupy r0-r5, pushing the call target into the
;; range where LowerKCFI_CHECK has to fall back off its default r6/r7
;; scratch pair.
define void @scratch_conflict(ptr noundef %fp, i32 %a, i32 %b, i32 %c,
                              i32 %d, i32 %e, i32 %f) {
  call void %fp(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f)
      [ "kcfi"(i32 12345678) ]
  ret void
}

;; Several checks in one function, so any per-function state in the lowering
;; has to be reset between them.
define void @repeated(ptr noundef %f, ptr noundef %g, ptr noundef %h) {
  call void %f() [ "kcfi"(i32 1) ]
  call void %g() [ "kcfi"(i32 12345678) ]
  call void %h() [ "kcfi"(i32 -1) ]
  ret void
}

;; Tail position: the call is the last thing in the function.
define void @tail(ptr noundef %fp) {
  tail call void %fp() [ "kcfi"(i32 4321) ]
  ret void
}

;; A noreturn target -- no return path after the call.
define void @noreturn_target(ptr noundef %fp) {
  call void %fp() #0 [ "kcfi"(i32 555) ]
  unreachable
}

;; Check inside a loop body, next to the loop's own compare and branch.
define void @in_loop(ptr noundef %fp, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  call void %fp() [ "kcfi"(i32 12345678) ]
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

;; The prefix form, where the load offset is not the default -4.
define void @prefixed(ptr noundef %fp) #1 {
  call void %fp() [ "kcfi"(i32 12345678) ]
  ret void
}

attributes #0 = { noreturn }
attributes #1 = { "patchable-function-prefix"="3" }

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
