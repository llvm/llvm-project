; NOTE: Do not autogenerate
; The update tools can't understand this usage of split-file. (Maybe we should
; add llc flags to set the PIC metadata.)
; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: cat a.ll nopic.ll > b.ll
; RUN: cat a.ll pic.ll > c.ll
; RUN: llc < b.ll | FileCheck %s
; RUN: llc < c.ll -relocation-model=pic | FileCheck %s -check-prefix=PIC

;--- a.ll
target triple = "aarch64-unknown-linux-gnu"

define dso_local void @f(ptr noundef %g) #0 {
; CHECK: adrp x9, __stack_chk_guard
; CHECK: ldr w9, [x9, :lo12:__stack_chk_guard]
; CHECK: adrp x8, __stack_chk_guard
; CHECK: ldr w8, [x8, :lo12:__stack_chk_guard]

; PIC: adrp x9, :got:__stack_chk_guard
; PIC: ldr x9, [x9, :got_lo12:__stack_chk_guard]
; PIC: ldr w9, [x9]
; PIC: adrp x8, :got:__stack_chk_guard
; PIC: ldr x8, [x8, :got_lo12:__stack_chk_guard]
; PIC: ldr w8, [x8]
entry:
  %g.addr = alloca ptr, align 8
  %x = alloca [1000 x i8], align 1
  store ptr %g, ptr %g.addr, align 8
  %0 = load ptr, ptr %g.addr, align 8
  %arraydecay = getelementptr inbounds [1000 x i8], ptr %x, i64 0, i64 0
  call void %0(ptr noundef %arraydecay)
  ret void
}

attributes #0 = { ssp }

;--- nopic.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"stack-protector-guard-value-width", i32 4}

;--- pic.ll
!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"stack-protector-guard-value-width", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
