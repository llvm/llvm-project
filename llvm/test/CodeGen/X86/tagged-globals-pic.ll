; RUN: llc --relocation-model=pic < %s | FileCheck %s
; RUN: llc --relocation-model=pic --relax-elf-relocations --filetype=obj -o - < %s | llvm-objdump -d -r - | FileCheck %s --check-prefix=OBJ

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@global = external global i32
declare void @func()

define ptr @global_addr() #0 {
  ; CHECK-LABEL: global_addr:
  ; CHECK: movq global@GOTPCREL_NORELAX(%rip), %rax
  ; CHECK: retq

  ; OBJ-LABEL: <global_addr>:
  ; OBJ: movq (%rip),
  ; OBJ-NEXT: R_X86_64_GOTPCREL global

  ret ptr @global
}

define i32 @global_load() #0 {
  ; CHECK-LABEL: global_load:
  ; CHECK: movq global@GOTPCREL_NORELAX(%rip), [[REG:%r[0-9a-z]+]]
  ; CHECK: movl ([[REG]]), %eax
  ; CHECK: retq

  ; OBJ-LABEL: <global_load>:
  ; OBJ: movq (%rip),
  ; OBJ-NEXT: R_X86_64_GOTPCREL global

  %load = load i32, ptr @global
  ret i32 %load
}

define void @global_store() #0 {
  ; CHECK-LABEL: global_store:
  ; CHECK: movq global@GOTPCREL_NORELAX(%rip), [[REG:%r[0-9a-z]+]]
  ; CHECK: movl $0, ([[REG]])
  ; CHECK: retq

  ; OBJ-LABEL: <global_store>:
  ; OBJ: movq (%rip),
  ; OBJ-NEXT: R_X86_64_GOTPCREL global

  store i32 0, ptr @global
  ret void
}

define ptr @func_addr() #0 {
  ; CHECK-LABEL: func_addr:
  ; CHECK: movq func@GOTPCREL(%rip), %rax
  ; CHECK: retq

  ; OBJ-LABEL: <func_addr>:
  ; OBJ: movq (%rip),
  ; OBJ-NEXT: R_X86_64_REX_GOTPCRELX func

  ret ptr @func
}

; Jump tables shouldn't go through the GOT.
define i32 @jump_table(i32 %x) #0 {
  ; CHECK-LABEL: jump_table:
  ; CHECK-NOT: @GOT

  switch i32 %x, label %default [
    i32 0, label %1
    i32 1, label %2
    i32 2, label %3
    i32 3, label %4
  ]
1:
  ret i32 7
2:
  ret i32 42
3:
  ret i32 3
4:
  ret i32 8
default:
  ret i32 %x
}

attributes #0 = { "target-features"="+tagged-globals" }
