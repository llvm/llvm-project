; Intel chips with slow unaligned memory accesses

; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=pentium3      2>&1 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=pentium3m     2>&1 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=pentium-m     2>&1 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=pentium4      2>&1 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=pentium4m     2>&1 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=yonah         2>&1 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=prescott      2>&1 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=nocona        2>&1 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=core2         2>&1 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=penryn        2>&1 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=bonnell       2>&1 | FileCheck %s --check-prefix=SLOW

; Intel chips with fast unaligned memory accesses

; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=silvermont     2>&1 | FileCheck %s --check-prefix=FAST
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=nehalem        2>&1 | FileCheck %s --check-prefix=FAST
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=westmere       2>&1 | FileCheck %s --check-prefix=FAST
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=sandybridge    2>&1 | FileCheck %s --check-prefix=FAST
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=ivybridge      2>&1 | FileCheck %s --check-prefix=FAST
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=haswell        2>&1 | FileCheck %s --check-prefix=FAST
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=broadwell      2>&1 | FileCheck %s --check-prefix=FAST
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=knl            2>&1 | FileCheck %s --check-prefix=FAST
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=skylake-avx512 2>&1 | FileCheck %s --check-prefix=FAST

; AMD chips with slow unaligned memory accesses

; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=athlon-4      2>&1 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=athlon-xp     2>&1 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=k8            2>&1 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=opteron       2>&1 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=athlon64      2>&1 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=athlon-fx     2>&1 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=k8-sse3       2>&1 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=opteron-sse3  2>&1 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=athlon64-sse3 2>&1 | FileCheck %s --check-prefix=SLOW

; AMD chips with fast unaligned memory accesses

; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=amdfam10      2>&1 | FileCheck %s --check-prefix=FAST
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=barcelona     2>&1 | FileCheck %s --check-prefix=FAST
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=btver1        2>&1 | FileCheck %s --check-prefix=FAST
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=btver2        2>&1 | FileCheck %s --check-prefix=FAST
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=bdver1        2>&1 | FileCheck %s --check-prefix=FAST
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=bdver2        2>&1 | FileCheck %s --check-prefix=FAST
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=bdver3        2>&1 | FileCheck %s --check-prefix=FAST
; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=bdver4        2>&1 | FileCheck %s --check-prefix=FAST

; Other chips with slow unaligned memory accesses

; RUN: llc < %s -mtriple=i386-unknown-unknown -mcpu=c3-2          2>&1 | FileCheck %s --check-prefix=SLOW

; Verify that the slow/fast unaligned memory attribute is set correctly for each CPU model.
; Slow chips use 4-byte stores. Fast chips with SSE or later use something other than 4-byte stores.
; Chips that don't have SSE use 4-byte stores either way, so they're not tested.

; Also verify that SSE4.2 or SSE4a imply fast unaligned accesses.

; RUN: llc < %s -mtriple=i386-unknown-unknown -mattr=sse4.2       2>&1 | FileCheck %s --check-prefix=FAST
; RUN: llc < %s -mtriple=i386-unknown-unknown -mattr=sse4a        2>&1 | FileCheck %s --check-prefix=FAST

define void @store_zeros(i8* %a) {
; SLOW-NOT: not a recognized processor
; SLOW-LABEL: store_zeros:
; SLOW:       # BB#0:
; SLOW-NEXT:    movl
; SLOW-NEXT:    movl
; SLOW-NEXT:    movl
; SLOW-NEXT:    movl
; SLOW-NEXT:    movl
; SLOW-NEXT:    movl
; SLOW-NEXT:    movl
; SLOW-NEXT:    movl
; SLOW-NEXT:    movl
; SLOW-NEXT:    movl
; SLOW-NEXT:    movl
; SLOW-NEXT:    movl
; SLOW-NEXT:    movl
; SLOW-NEXT:    movl
; SLOW-NEXT:    movl
; SLOW-NEXT:    movl
; SLOW-NEXT:    movl
;
; FAST-NOT: not a recognized processor
; FAST-LABEL: store_zeros:
; FAST:       # BB#0:
; FAST-NEXT:    movl {{[0-9]+}}(%esp), %eax
; FAST-NOT:     movl
  call void @llvm.memset.p0i8.i64(i8* %a, i8 0, i64 64, i32 1, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1)

