; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32r2 \
; RUN:     < %s -verify-machineinstrs | FileCheck %s

%struct.x = type { i32 }

@i = common global i32 0, align 4

define i32 @foobar(i32 signext %x) {
entry:
; CHECK-LABEL: foobar:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %a = alloca %struct.x, align 8
  %c = alloca ptr, align 8
  store i32 %x, ptr %x.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  store i32 %0, ptr %a, align 4
  store ptr %a, ptr %c, align 4
  %1 = load ptr, ptr %c, align 4
  %2 = load i32, ptr %1, align 4
  store i32 %2, ptr @i, align 4
  %3 = load i32, ptr %retval
; CHECK:        addiu   $[[A_ADDR:[0-9]+]], $sp, 8
; CHECK-DAG:    lw      $[[I_ADDR:[0-9]+]], %got(i)($[[REG_GP:[0-9]+]])
; CHECK-DAG:    sw      $[[A_ADDR]], [[A_ADDR_FI:[0-9]+]]($sp)
; CHECK-DAG:    lw      $[[A_ADDR2:[0-9]+]], [[A_ADDR_FI]]($sp)
; CHECK-DAG:    lw      $[[A_X:[0-9]+]], 0($[[A_ADDR2]])
; CHECK-DAG:    sw      $[[A_X]], 0($[[I_ADDR]])
  ret i32 %3
}
