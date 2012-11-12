; RUN: llc -march=x86 -mattr=+cmov -mtriple=i386-pc-linux < %s | FileCheck %s -check-prefix=LINUX
; RUN: llc -march=x86 -mtriple=i386-macosx -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC

@sc64 = external global i64

define void @atomic_maxmin_i6432() {
; LINUX: atomic_maxmin_i6432
  %1 = atomicrmw max  i64* @sc64, i64 5 acquire
; LINUX: [[LABEL:.LBB[0-9]+_[0-9]+]]
; LINUX: cmpl
; LINUX: setl
; LINUX: cmpl
; LINUX: setl
; LINUX: cmovne
; LINUX: cmovne
; LINUX: lock
; LINUX-NEXT: cmpxchg8b
; LINUX: jne [[LABEL]]
  %2 = atomicrmw min  i64* @sc64, i64 6 acquire
; LINUX: [[LABEL:.LBB[0-9]+_[0-9]+]]
; LINUX: cmpl
; LINUX: setg
; LINUX: cmpl
; LINUX: setg
; LINUX: cmovne
; LINUX: cmovne
; LINUX: lock
; LINUX-NEXT: cmpxchg8b
; LINUX: jne [[LABEL]]
  %3 = atomicrmw umax i64* @sc64, i64 7 acquire
; LINUX: [[LABEL:.LBB[0-9]+_[0-9]+]]
; LINUX: cmpl
; LINUX: setb
; LINUX: cmpl
; LINUX: setb
; LINUX: cmovne
; LINUX: cmovne
; LINUX: lock
; LINUX-NEXT: cmpxchg8b
; LINUX: jne [[LABEL]]
  %4 = atomicrmw umin i64* @sc64, i64 8 acquire
; LINUX: [[LABEL:.LBB[0-9]+_[0-9]+]]
; LINUX: cmpl
; LINUX: seta
; LINUX: cmpl
; LINUX: seta
; LINUX: cmovne
; LINUX: cmovne
; LINUX: lock
; LINUX-NEXT: cmpxchg8b
; LINUX: jne [[LABEL]]
  ret void
}

; rdar://12453106
@id = internal global i64 0, align 8

define void @tf_bug(i8* %ptr) nounwind {
; PIC: tf_bug:
; PIC: movl _id-L1$pb(
; PIC: movl (_id-L1$pb)+4(
  %tmp1 = atomicrmw add i64* @id, i64 1 seq_cst
  %tmp2 = add i64 %tmp1, 1
  %tmp3 = bitcast i8* %ptr to i64*
  store i64 %tmp2, i64* %tmp3, align 4
  ret void
}
