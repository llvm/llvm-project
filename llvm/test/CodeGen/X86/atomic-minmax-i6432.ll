; RUN: llc -march=x86 -mattr=+cmov,cx16 -mtriple=i386-pc-linux -verify-machineinstrs < %s | FileCheck %s -check-prefix=LINUX
; RUN: llc -march=x86 -mattr=cx16 -mtriple=i386-macosx -relocation-model=pic -verify-machineinstrs < %s | FileCheck %s -check-prefix=PIC

@sc64 = external global i64

define void @atomic_maxmin_i6432() {
; LINUX: atomic_maxmin_i6432
  %1 = atomicrmw max  i64* @sc64, i64 5 acquire
; LINUX: [[LABEL:.LBB[0-9]+_[0-9]+]]
; LINUX: cmpl
; LINUX: seta
; LINUX: cmovne
; LINUX: cmovne
; LINUX: lock cmpxchg8b
; LINUX: jne [[LABEL]]
  %2 = atomicrmw min  i64* @sc64, i64 6 acquire
; LINUX: [[LABEL:.LBB[0-9]+_[0-9]+]]
; LINUX: cmpl
; LINUX: setb
; LINUX: cmovne
; LINUX: cmovne
; LINUX: lock cmpxchg8b
; LINUX: jne [[LABEL]]
  %3 = atomicrmw umax i64* @sc64, i64 7 acquire
; LINUX: [[LABEL:.LBB[0-9]+_[0-9]+]]
; LINUX: cmpl
; LINUX: seta
; LINUX: cmovne
; LINUX: cmovne
; LINUX: lock cmpxchg8b
; LINUX: jne [[LABEL]]
  %4 = atomicrmw umin i64* @sc64, i64 8 acquire
; LINUX: [[LABEL:.LBB[0-9]+_[0-9]+]]
; LINUX: cmpl
; LINUX: setb
; LINUX: cmovne
; LINUX: cmovne
; LINUX: lock cmpxchg8b
; LINUX: jne [[LABEL]]
  ret void
}

; rdar://12453106
@id = internal global i64 0, align 8

define void @tf_bug(i8* %ptr) nounwind {
; PIC-LABEL: tf_bug:
; PIC-DAG: movl _id-L1$pb(
; PIC-DAG: movl (_id-L1$pb)+4(
  %tmp1 = atomicrmw add i64* @id, i64 1 seq_cst
  %tmp2 = add i64 %tmp1, 1
  %tmp3 = bitcast i8* %ptr to i64*
  store i64 %tmp2, i64* %tmp3, align 4
  ret void
}
