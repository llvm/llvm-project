; RUN: llc -mtriple=aarch64-pc-windows-msvc -O1 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-pc-windows-msvc -O3 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-pc-windows-msvc -verify-machineinstrs -filetype=obj -o %t < %s
; RUN: llvm-objdump -d %t | FileCheck %s --check-prefix=OBJ

; OBJ-LABEL: <empty_entry>:
; OBJ-NEXT:  {{[0-9a-f]+}}: {{[0-9a-f]+}} nop
; OBJ-NEXT:  {{[0-9a-f]+}}: {{[0-9a-f]+}} ldr w8, [x0]
; OBJ:       b.lo 0x4 <empty_entry+0x4>

; The entry block becomes empty after register allocation. The loop backedge
; must skip the instruction that will be overwritten by a hotpatch.
define void @empty_entry(ptr %a, ptr %b) #0 {
; CHECK-LABEL: empty_entry:
; CHECK:       // %bb.0:
; CHECK-NEXT:  nop
; CHECK-NEXT:  [[LOOP:\.LBB[0-9]+_[0-9]+]]:
; CHECK:       ldr w8, [x0]
; CHECK:       b.lo [[LOOP]]
entry:
  br label %loop
loop:
  %p = phi ptr [ %a, %entry ], [ %next, %loop ]
  %value = load i32, ptr %p
  %inc = add i32 %value, 1
  store i32 %inc, ptr %p
  %next = getelementptr i32, ptr %p, i64 1
  %again = icmp ult ptr %next, %b
  br i1 %again, label %loop, label %exit
exit:
  ret void
}

; An entry block that already emits an instruction needs no padding.
define void @nonempty_entry(ptr %a, ptr %b) #0 {
; CHECK-LABEL: nonempty_entry:
; CHECK-NOT:   nop
; CHECK:       ldr w8, [x0]
; CHECK-NOT:   nop
; CHECK:       [[LOOP:\.LBB[0-9]+_[0-9]+]]:
; CHECK:       b.lo [[LOOP]]
entry:
  %initial = load i32, ptr %a
  br label %loop
loop:
  %p = phi ptr [ %a, %entry ], [ %next, %loop ]
  store i32 %initial, ptr %p
  %next = getelementptr i32, ptr %p, i64 1
  %again = icmp ult ptr %next, %b
  br i1 %again, label %loop, label %exit
exit:
  ret void
}

define void @return_only() #0 {
; CHECK-LABEL: return_only:
; CHECK:       // %bb.0:
; CHECK-NEXT:  ret
  ret void
}

; Inline assembly may emit no instructions, or define a label at its first
; instruction. Conservatively pad an entry that begins with inline assembly.
define void @inline_asm() #0 {
; CHECK-LABEL: inline_asm:
; CHECK:       // %bb.0:
; CHECK-NEXT:  nop
; CHECK-NEXT:  //APP
; CHECK:       //NO_APP
; CHECK-NEXT:  ret
  call void asm sideeffect "", ""()
  ret void
}

; A zero-length patchable-function-entry must not suppress hotpatch padding.
define void @zero_patchable_entry() #1 {
; CHECK-LABEL: zero_patchable_entry:
; CHECK:       // %bb.0:
; CHECK-NEXT:  nop
; CHECK-NEXT:  [[LOOP:\.LBB[0-9]+_[0-9]+]]:
; CHECK:       b [[LOOP]]
entry:
  br label %loop
loop:
  br label %loop
}

; Existing patchable entry padding already separates the function entry from
; the loop header, so it must not acquire a second NOP.
define void @padded_entry() #2 {
; CHECK-LABEL: padded_entry:
; CHECK:       // %bb.0:
; CHECK-NEXT:  nop
; CHECK-NEXT:  [[LOOP:\.LBB[0-9]+_[0-9]+]]:
; CHECK:       b [[LOOP]]
entry:
  br label %loop
loop:
  br label %loop
}

; Functions without the hotpatch attribute retain their original layout.
define void @unpatched_entry() {
; CHECK-LABEL: unpatched_entry:
; CHECK:       // %bb.0:
; CHECK-NEXT:  [[LOOP:\.LBB[0-9]+_[0-9]+]]:
; CHECK:       b [[LOOP]]
entry:
  br label %loop
loop:
  br label %loop
}

attributes #0 = { "patchable-function"="prologue-short-redirect" }
attributes #1 = { "patchable-function"="prologue-short-redirect" "patchable-function-entry"="0" }
attributes #2 = { "patchable-function"="prologue-short-redirect" "patchable-function-entry"="1" }
