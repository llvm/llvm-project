; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32r2 \
; RUN:     < %s -verify-machineinstrs | FileCheck %s

%struct.x = type { i32 }

@i = common global i32 0, align 4

define i32 @foobar(i32 signext %x) {
entry:
; CHECK-LABEL: foobar:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %a = alloca %struct.x, align 4
  %c = alloca %struct.x*, align 4
  store i32 %x, i32* %x.addr, align 4
  %x1 = getelementptr inbounds %struct.x, %struct.x* %a, i32 0, i32 0
  %0 = load i32, i32* %x.addr, align 4
  store i32 %0, i32* %x1, align 4
  store %struct.x* %a, %struct.x** %c, align 4
  %1 = load %struct.x*, %struct.x** %c, align 4
  %x2 = getelementptr inbounds %struct.x, %struct.x* %1, i32 0, i32 0
  %2 = load i32, i32* %x2, align 4
  store i32 %2, i32* @i, align 4
  %3 = load i32, i32* %retval
; CHECK-DAG:    lw      $[[I_ADDR:[0-9]+]], %got(i)($[[REG_GP:[0-9]+]])
; CHECK-DAG:    addiu   $[[A_ADDR:[0-9]+]], $sp, 8
; CHECK-DAG:    sw      $[[A_ADDR]], [[A_ADDR_FI:[0-9]+]]($sp)
; CHECK-DAG:    lw      $[[A_ADDR2:[0-9]+]], [[A_ADDR_FI]]($sp)
; CHECK-DAG:    lw      $[[A_X:[0-9]+]], 0($[[A_ADDR2]])
; CHECK-DAG:    sw      $[[A_X]], 0($[[I_ADDR]])
  ret i32 %3
}
