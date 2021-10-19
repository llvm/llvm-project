; RUN: llc -verify-machineinstrs -mtriple=x86_64-unknown -mattr=+lvi-cfi < %s | FileCheck %s --check-prefix=X64
; RUN: llc -verify-machineinstrs -mtriple=x86_64-unknown -mattr=+lvi-cfi -O0 < %s | FileCheck %s --check-prefix=X64FAST
;
; Note that a lot of this code was lifted from retpoline.ll.

declare dso_local void @bar(i32)

; Test a simple indirect call and tail call.
define void @icall_reg(void (i32)* %fp, i32 %x) {
entry:
  tail call void @bar(i32 %x)
  tail call void %fp(i32 %x)
  tail call void @bar(i32 %x)
  tail call void %fp(i32 %x)
  ret void
}

; X64-LABEL: icall_reg:
; X64-DAG:   movq %rdi, %[[fp:[^ ]*]]
; X64-DAG:   movl %esi, %[[x:[^ ]*]]
; X64:       movl %esi, %edi
; X64:       callq bar
; X64-DAG:   movl %[[x]], %edi
; X64-DAG:   movq %[[fp]], %r11
; X64:       callq __llvm_lvi_thunk_r11
; X64:       movl %[[x]], %edi
; X64:       callq bar
; X64-DAG:   movl %[[x]], %edi
; X64-DAG:   movq %[[fp]], %r11
; X64:       jmp __llvm_lvi_thunk_r11 # TAILCALL

; X64FAST-LABEL: icall_reg:
; X64FAST:       callq bar
; X64FAST:       callq __llvm_lvi_thunk_r11
; X64FAST:       callq bar
; X64FAST:       jmp __llvm_lvi_thunk_r11 # TAILCALL


@global_fp = external dso_local global void (i32)*

; Test an indirect call through a global variable.
define void @icall_global_fp(i32 %x, void (i32)** %fpp) #0 {
  %fp1 = load void (i32)*, void (i32)** @global_fp
  call void %fp1(i32 %x)
  %fp2 = load void (i32)*, void (i32)** @global_fp
  tail call void %fp2(i32 %x)
  ret void
}

; X64-LABEL: icall_global_fp:
; X64-DAG:   movl %edi, %[[x:[^ ]*]]
; X64-DAG:   movq global_fp(%rip), %r11
; X64:       callq __llvm_lvi_thunk_r11
; X64-DAG:   movl %[[x]], %edi
; X64-DAG:   movq global_fp(%rip), %r11
; X64:       jmp __llvm_lvi_thunk_r11 # TAILCALL

; X64FAST-LABEL: icall_global_fp:
; X64FAST:       movq global_fp(%rip), %r11
; X64FAST:       callq __llvm_lvi_thunk_r11
; X64FAST:       movq global_fp(%rip), %r11
; X64FAST:       jmp __llvm_lvi_thunk_r11 # TAILCALL


%struct.Foo = type { void (%struct.Foo*)** }

; Test an indirect call through a vtable.
define void @vcall(%struct.Foo* %obj) #0 {
  %vptr_field = getelementptr %struct.Foo, %struct.Foo* %obj, i32 0, i32 0
  %vptr = load void (%struct.Foo*)**, void (%struct.Foo*)*** %vptr_field
  %vslot = getelementptr void(%struct.Foo*)*, void(%struct.Foo*)** %vptr, i32 1
  %fp = load void(%struct.Foo*)*, void(%struct.Foo*)** %vslot
  tail call void %fp(%struct.Foo* %obj)
  tail call void %fp(%struct.Foo* %obj)
  ret void
}

; X64-LABEL: vcall:
; X64:       movq %rdi, %[[obj:[^ ]*]]
; X64:       movq (%rdi), %[[vptr:[^ ]*]]
; X64:       movq 8(%[[vptr]]), %[[fp:[^ ]*]]
; X64:       movq %[[fp]], %r11
; X64:       callq __llvm_lvi_thunk_r11
; X64-DAG:   movq %[[obj]], %rdi
; X64-DAG:   movq %[[fp]], %r11
; X64:       jmp __llvm_lvi_thunk_r11 # TAILCALL

; X64FAST-LABEL: vcall:
; X64FAST:       callq __llvm_lvi_thunk_r11
; X64FAST:       jmp __llvm_lvi_thunk_r11 # TAILCALL


declare dso_local void @direct_callee()

define void @direct_tail() #0 {
  tail call void @direct_callee()
  ret void
}

; X64-LABEL: direct_tail:
; X64:       jmp direct_callee # TAILCALL
; X64FAST-LABEL: direct_tail:
; X64FAST:   jmp direct_callee # TAILCALL


declare void @nonlazybind_callee() #1

define void @nonlazybind_caller() #0 {
  call void @nonlazybind_callee()
  tail call void @nonlazybind_callee()
  ret void
}

; X64-LABEL: nonlazybind_caller:
; X64:       movq nonlazybind_callee@GOTPCREL(%rip), %[[REG:.*]]
; X64:       movq %[[REG]], %r11
; X64:       callq __llvm_lvi_thunk_r11
; X64:       movq %[[REG]], %r11
; X64:       jmp __llvm_lvi_thunk_r11 # TAILCALL
; X64FAST-LABEL: nonlazybind_caller:
; X64FAST:   movq nonlazybind_callee@GOTPCREL(%rip), %r11
; X64FAST:   callq __llvm_lvi_thunk_r11
; X64FAST:   movq nonlazybind_callee@GOTPCREL(%rip), %r11
; X64FAST:   jmp __llvm_lvi_thunk_r11 # TAILCALL


; Check that a switch gets lowered using a jump table
define void @switch_jumptable(i32* %ptr, i64* %sink) #0 {
; X64-LABEL: switch_jumptable:
; X64_NOT:      jmpq *
entry:
  br label %header

header:
  %i = load volatile i32, i32* %ptr
  switch i32 %i, label %bb0 [
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
    i32 5, label %bb5
    i32 6, label %bb6
    i32 7, label %bb7
    i32 8, label %bb8
    i32 9, label %bb9
  ]

bb0:
  store volatile i64 0, i64* %sink
  br label %header

bb1:
  store volatile i64 1, i64* %sink
  br label %header

bb2:
  store volatile i64 2, i64* %sink
  br label %header

bb3:
  store volatile i64 3, i64* %sink
  br label %header

bb4:
  store volatile i64 4, i64* %sink
  br label %header

bb5:
  store volatile i64 5, i64* %sink
  br label %header

bb6:
  store volatile i64 6, i64* %sink
  br label %header

bb7:
  store volatile i64 7, i64* %sink
  br label %header

bb8:
  store volatile i64 8, i64* %sink
  br label %header

bb9:
  store volatile i64 9, i64* %sink
  br label %header
}


@indirectbr_rewrite.targets = constant [10 x i8*] [i8* blockaddress(@indirectbr_rewrite, %bb0),
                                                   i8* blockaddress(@indirectbr_rewrite, %bb1),
                                                   i8* blockaddress(@indirectbr_rewrite, %bb2),
                                                   i8* blockaddress(@indirectbr_rewrite, %bb3),
                                                   i8* blockaddress(@indirectbr_rewrite, %bb4),
                                                   i8* blockaddress(@indirectbr_rewrite, %bb5),
                                                   i8* blockaddress(@indirectbr_rewrite, %bb6),
                                                   i8* blockaddress(@indirectbr_rewrite, %bb7),
                                                   i8* blockaddress(@indirectbr_rewrite, %bb8),
                                                   i8* blockaddress(@indirectbr_rewrite, %bb9)]

; Check that when thunks are enabled the indirectbr instruction gets
; rewritten to use switch, and that in turn doesn't get lowered as a jump
; table.
define void @indirectbr_rewrite(i64* readonly %p, i64* %sink) #0 {
; X64-LABEL: indirectbr_rewrite:
; X64-NOT:     jmpq *
entry:
  %i0 = load i64, i64* %p
  %target.i0 = getelementptr [10 x i8*], [10 x i8*]* @indirectbr_rewrite.targets, i64 0, i64 %i0
  %target0 = load i8*, i8** %target.i0
  indirectbr i8* %target0, [label %bb1, label %bb3]

bb0:
  store volatile i64 0, i64* %sink
  br label %latch

bb1:
  store volatile i64 1, i64* %sink
  br label %latch

bb2:
  store volatile i64 2, i64* %sink
  br label %latch

bb3:
  store volatile i64 3, i64* %sink
  br label %latch

bb4:
  store volatile i64 4, i64* %sink
  br label %latch

bb5:
  store volatile i64 5, i64* %sink
  br label %latch

bb6:
  store volatile i64 6, i64* %sink
  br label %latch

bb7:
  store volatile i64 7, i64* %sink
  br label %latch

bb8:
  store volatile i64 8, i64* %sink
  br label %latch

bb9:
  store volatile i64 9, i64* %sink
  br label %latch

latch:
  %i.next = load i64, i64* %p
  %target.i.next = getelementptr [10 x i8*], [10 x i8*]* @indirectbr_rewrite.targets, i64 0, i64 %i.next
  %target.next = load i8*, i8** %target.i.next
  ; Potentially hit a full 10 successors here so that even if we rewrite as
  ; a switch it will try to be lowered with a jump table.
  indirectbr i8* %target.next, [label %bb0,
                                label %bb1,
                                label %bb2,
                                label %bb3,
                                label %bb4,
                                label %bb5,
                                label %bb6,
                                label %bb7,
                                label %bb8,
                                label %bb9]
}

; Lastly check that the necessary thunks were emitted.
;
; X64-LABEL:         .section        .text.__llvm_lvi_thunk_r11,{{.*}},__llvm_lvi_thunk_r11,comdat
; X64-NEXT:          .hidden __llvm_lvi_thunk_r11
; X64-NEXT:          .weak   __llvm_lvi_thunk_r11
; X64:       __llvm_lvi_thunk_r11:
; X64-NEXT:  # {{.*}}                                # %entry
; X64-NEXT:          lfence
; X64-NEXT:          jmpq     *%r11

attributes #1 = { nonlazybind }
