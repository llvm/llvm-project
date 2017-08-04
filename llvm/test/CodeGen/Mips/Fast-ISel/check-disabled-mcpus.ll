; Targets where we should not enable FastISel.
; RUN: llc -march=mips -mcpu=mips2 -O0 -relocation-model=pic \
; RUN:     -pass-remarks-missed=isel <%s 2>&1 | FileCheck %s
; RUN: llc -march=mips -mcpu=mips3 -O0 -relocation-model=pic -target-abi n64 \
; RUN:     -pass-remarks-missed=isel <%s 2>&1 | FileCheck %s
; RUN: llc -march=mips -mcpu=mips4 -O0 -relocation-model=pic -target-abi n64 \
; RUN:     -pass-remarks-missed=isel <%s 2>&1 | FileCheck %s

; RUN: llc -march=mips -mcpu=mips32r6 -O0 -relocation-model=pic \
; RUN:     -pass-remarks-missed=isel <%s 2>&1 | FileCheck %s

; RUN: llc -march=mips -mattr=mips16 -O0 -relocation-model=pic \
; RUN:     -pass-remarks-missed=isel <%s 2>&1 | FileCheck %s

; RUN: llc -march=mips -mcpu=mips32r2 -mattr=+micromips -O0 -relocation-model=pic \
; RUN:     -pass-remarks-missed=isel <%s 2>&1 | FileCheck %s
; RUN: llc -march=mips -mcpu=mips32r3 -mattr=+micromips -O0 -relocation-model=pic \
; RUN:     -pass-remarks-missed=isel <%s 2>&1 | FileCheck %s
; RUN: llc -march=mips -mcpu=mips32r5 -mattr=+micromips -O0 -relocation-model=pic \
; RUN:     -pass-remarks-missed=isel <%s 2>&1 | FileCheck %s

; RUN: llc -march=mips -mcpu=mips64 -O0 -relocation-model=pic -target-abi n64 \
; RUN:     -pass-remarks-missed=isel <%s 2>&1 | FileCheck %s
; RUN: llc -march=mips -mcpu=mips64r2 -O0 -relocation-model=pic -target-abi n64 \
; RUN:     -pass-remarks-missed=isel <%s 2>&1 | FileCheck %s
; RUN: llc -march=mips -mcpu=mips64r3 -O0 -relocation-model=pic -target-abi n64 \
; RUN:     -pass-remarks-missed=isel <%s 2>&1 | FileCheck %s
; RUN: llc -march=mips -mcpu=mips64r5 -O0 -relocation-model=pic -target-abi n64 \
; RUN:     -pass-remarks-missed=isel <%s 2>&1 | FileCheck %s
; RUN: llc -march=mips -mcpu=mips32r6 -O0 -relocation-model=pic \
; RUN:     -pass-remarks-missed=isel <%s 2>&1 | FileCheck %s

; Valid targets for FastISel.
; RUN: llc -march=mips -mcpu=mips32r0 -O0 -relocation-model=pic \
; RUN:     -pass-remarks-missed=isel <%s 2>&1 | FileCheck %s -check-prefix=FISEL
; RUN: llc -march=mips -mcpu=mips32r2 -O0 -relocation-model=pic \
; RUN:     -pass-remarks-missed=isel <%s 2>&1 | FileCheck %s -check-prefix=FISEL

; The CHECK prefix is being used by those targets that do not support FastISel.
; By checking that we don't emit the "FastISel missed terminator..." message,
; we ensure that we do not generate code through FastISel.

; CHECK-NOT: FastISel missed terminator:   ret i64 0

; The above CHECK will only be valid as long as we *do* emit the missed
; terminator message for targets that support FastISel. If we add support
; for i64 return values in the future, then the following FISEL check-prefix
; will fail and we will have to come up with a new test.

; FISEL: FastISel missed terminator:   ret i64 0

define i64 @foo() {
entry:
  ret i64 0
}
