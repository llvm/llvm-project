; RUN: llc -march=mips -mcpu=mips2 -O0 -relocation-model=pic \
; RUN:     -fast-isel-verbose <%s 2>&1 | FileCheck %s
; RUN: llc -march=mips -mcpu=mips3 -O0 -relocation-model=pic -target-abi n64 \
; RUN:     -fast-isel-verbose <%s 2>&1 | FileCheck %s
; RUN: llc -march=mips -mcpu=mips4 -O0 -relocation-model=pic -target-abi n64 \
; RUN:     -fast-isel-verbose <%s 2>&1 | FileCheck %s

; RUN: llc -march=mips -mcpu=mips32r6 -O0 -relocation-model=pic \
; RUN:     -fast-isel-verbose <%s 2>&1 | FileCheck %s
; RUN: llc -march=mips -mcpu=mips32r2 -mattr=+micromips -O0 -relocation-model=pic \
; RUN:     -fast-isel-verbose <%s 2>&1 | FileCheck %s

; RUN: llc -march=mips -mcpu=mips64 -O0 -relocation-model=pic -target-abi n64 \
; RUN:     -fast-isel-verbose <%s 2>&1 | FileCheck %s
; RUN: llc -march=mips -mcpu=mips64r2 -O0 -relocation-model=pic -target-abi n64 \
; RUN:     -fast-isel-verbose <%s 2>&1 | FileCheck %s
; RUN: llc -march=mips -mcpu=mips64r3 -O0 -relocation-model=pic -target-abi n64 \
; RUN:     -fast-isel-verbose <%s 2>&1 | FileCheck %s
; RUN: llc -march=mips -mcpu=mips64r5 -O0 -relocation-model=pic -target-abi n64 \
; RUN:     -fast-isel-verbose <%s 2>&1 | FileCheck %s
; RUN: llc -march=mips -mcpu=mips32r6 -O0 -relocation-model=pic \
; RUN:     -fast-isel-verbose <%s 2>&1 | FileCheck %s

; CHECK: FastISel missed terminator:   ret i32 0

define i32 @foo() {
entry:
  ret i32 0
}
