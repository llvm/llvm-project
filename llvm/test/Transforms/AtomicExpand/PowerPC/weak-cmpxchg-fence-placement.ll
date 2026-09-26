; RUN: opt -S -mtriple=powerpc-unknown-linux-musl \
; RUN:   -passes='require<libcall-lowering-info>,atomic-expand' %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,HOIST
; RUN: opt -S -mtriple=powerpc-unknown-linux-musl -mcpu=e500 \
; RUN:   -passes='require<libcall-lowering-info>,atomic-expand' %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,HOIST
; RUN: opt -S -mtriple=powerpc-unknown-linux-musl \
; RUN:   -mattr=+fence-keeps-reservation \
; RUN:   -passes='require<libcall-lowering-info>,atomic-expand' %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,SINK
; RUN: opt -S -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -passes='require<libcall-lowering-info>,atomic-expand' %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,SINK

; A weak cmpxchg gets its release fence before the lwarx (HOIST) unless the CPU
; has fence-keeps-reservation (SINK). A strong cmpxchg always sinks it.

define i1 @weak_release(ptr %p) {
; CHECK-LABEL: define i1 @weak_release(
; HOIST-NEXT:    call void @llvm.ppc.lwsync()
; SINK-NOT:      @llvm.ppc.{{(lw)?}}sync
; CHECK:         call i32 @llvm.ppc.lwarx(ptr %p)
; HOIST-NOT:     @llvm.ppc.{{(lw)?}}sync
; SINK:        cmpxchg.fencedstore:
; SINK-NEXT:     call void @llvm.ppc.lwsync()
; CHECK:         call i32 @llvm.ppc.stwcx(ptr %p, i32 1)
  %pair = cmpxchg weak ptr %p, i32 0, i32 1 release monotonic
  %ok = extractvalue { i32, i1 } %pair, 1
  ret i1 %ok
}

define i1 @weak_seq_cst(ptr %p) {
; CHECK-LABEL: define i1 @weak_seq_cst(
; HOIST-NEXT:    call void @llvm.ppc.sync()
; SINK-NOT:      @llvm.ppc.{{(lw)?}}sync
; CHECK:         call i32 @llvm.ppc.lwarx(ptr %p)
; HOIST-NOT:     @llvm.ppc.{{(lw)?}}sync
; SINK:        cmpxchg.fencedstore:
; SINK-NEXT:     call void @llvm.ppc.sync()
; CHECK:         call i32 @llvm.ppc.stwcx(ptr %p, i32 1)
  %pair = cmpxchg weak ptr %p, i32 0, i32 1 seq_cst seq_cst
  %ok = extractvalue { i32, i1 } %pair, 1
  ret i1 %ok
}

define i1 @strong_release(ptr %p) {
; CHECK-LABEL: define i1 @strong_release(
; CHECK-NOT:     @llvm.ppc.{{(lw)?}}sync
; CHECK:         call i32 @llvm.ppc.lwarx(ptr %p)
; CHECK:       cmpxchg.fencedstore:
; CHECK-NEXT:    call void @llvm.ppc.lwsync()
; CHECK:         call i32 @llvm.ppc.stwcx(ptr %p, i32 1)
  %pair = cmpxchg ptr %p, i32 0, i32 1 release monotonic
  %ok = extractvalue { i32, i1 } %pair, 1
  ret i1 %ok
}
