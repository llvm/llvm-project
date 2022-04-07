; Check that the CPU names work.

; RUN: llc -mtriple=mips -mcpu=generic -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefix=GENERIC
; GENERIC: ISA: MIPS32

; RUN: llc -mtriple=mips -mcpu=mips1 -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefix=MIPS1
; MIPS1: ISA: MIPS1
; RUN: llc -mtriple=mips -mcpu=mips2 -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefix=MIPS2
; MIPS2: ISA: MIPS2
; RUN: llc -mtriple=mips64 -mcpu=mips3 -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefix=MIPS3
; MIPS3: ISA: MIPS3
; RUN: llc -mtriple=mips64 -mcpu=mips4 -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefix=MIPS4
; MIPS4: ISA: MIPS4

; RUN: llc -mtriple=mips -mcpu=mips32 -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefix=MIPS32
; MIPS32: ISA: MIPS32
; RUN: llc -mtriple=mips -mcpu=mips32r2 -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefix=MIPS32R2
; MIPS32R2: ISA: MIPS32r2
; RUN: llc -mtriple=mips -mcpu=mips32r3 -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefix=MIPS32R3
; MIPS32R3: ISA: MIPS32r3
; RUN: llc -mtriple=mips -mcpu=mips32r5 -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefix=MIPS32R5
; MIPS32R5: ISA: MIPS32r5
; RUN: llc -mtriple=mips -mcpu=mips32r6 -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefix=MIPS32R6
; MIPS32R6: ISA: MIPS32r6

; RUN: llc -mtriple=mips64 -mcpu=mips64 -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefix=MIPS64
; MIPS64: ISA: MIPS64
; RUN: llc -mtriple=mips64 -mcpu=mips64r2 -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefix=MIPS64R2
; MIPS64R2: ISA: MIPS64r2
; RUN: llc -mtriple=mips64 -mcpu=mips64r3 -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefix=MIPS64R3
; MIPS64R3: ISA: MIPS64r3
; RUN: llc -mtriple=mips64 -mcpu=mips64r5 -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefix=MIPS64R5
; MIPS64R5: ISA: MIPS64r5
; RUN: llc -mtriple=mips64 -mcpu=mips64r6 -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefix=MIPS64R6
; MIPS64R6: ISA: MIPS64r6

; RUN: llc -mtriple=mips64 -mcpu=octeon -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefix=OCTEON
; OCTEON: ISA: MIPS64r2
; OCTEON: ISA Extension: Cavium Networks Octeon
; RUN: llc -mtriple=mips64 -mcpu=octeon+ -filetype=obj < %s \
; RUN:   | llvm-readelf -A - | FileCheck %s --check-prefix=OCTEONP
; OCTEONP: ISA: MIPS64r2
; OCTEONP: ISA Extension: Cavium Networks OcteonP

; Check that we reject CPUs that are not implemented.

; RUN: not --crash llc < %s -o /dev/null -mtriple=mips64 -mcpu=mips5 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ERROR

; ERROR: LLVM ERROR: Code generation for MIPS-{{.}} is not implemented

define void @foo() {
  ret void
}
