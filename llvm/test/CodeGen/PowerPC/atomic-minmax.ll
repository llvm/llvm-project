; RUN: llc < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define void @a32min(i32* nocapture dereferenceable(4) %minimum, i32 %val) #0 {
entry:
  %0 = atomicrmw min i32* %minimum, i32 %val monotonic
  ret void

; CHECK-LABEL: @a32min
; CHECK: lwarx [[OLDV:[0-9]+]], 0, 3
; CHECK: cmpw 4, [[OLDV]]
; CHECK: bgelr 0
; CHECK: stwcx. 4, 0, 3
; CHECK: bne 0,
; CHECK: blr
}

define void @a32max(i32* nocapture dereferenceable(4) %minimum, i32 %val) #0 {
entry:
  %0 = atomicrmw max i32* %minimum, i32 %val monotonic
  ret void

; CHECK-LABEL: @a32max
; CHECK: lwarx [[OLDV:[0-9]+]], 0, 3
; CHECK: cmpw 4, [[OLDV]]
; CHECK: blelr 0
; CHECK: stwcx. 4, 0, 3
; CHECK: bne 0,
; CHECK: blr
}

define void @a32umin(i32* nocapture dereferenceable(4) %minimum, i32 %val) #0 {
entry:
  %0 = atomicrmw umin i32* %minimum, i32 %val monotonic
  ret void

; CHECK-LABEL: @a32umin
; CHECK: lwarx [[OLDV:[0-9]+]], 0, 3
; CHECK: cmplw 4, [[OLDV]]
; CHECK: bgelr 0
; CHECK: stwcx. 4, 0, 3
; CHECK: bne 0,
; CHECK: blr
}

define void @a32umax(i32* nocapture dereferenceable(4) %minimum, i32 %val) #0 {
entry:
  %0 = atomicrmw umax i32* %minimum, i32 %val monotonic
  ret void

; CHECK-LABEL: @a32umax
; CHECK: lwarx [[OLDV:[0-9]+]], 0, 3
; CHECK: cmplw 4, [[OLDV]]
; CHECK: blelr 0
; CHECK: stwcx. 4, 0, 3
; CHECK: bne 0,
; CHECK: blr
}

define void @a16min(i16* nocapture dereferenceable(4) %minimum, i16 %val) #1 {
entry:
  %0 = atomicrmw min i16* %minimum, i16 %val monotonic
  ret void

; CHECK-LABEL: @a16min
; CHECK: lharx [[OLDV:[0-9]+]], 0, 3
; CHECK: cmpw 4, [[OLDV]]
; CHECK: bgelr 0
; CHECK: sthcx. 4, 0, 3
; CHECK: bne 0,
; CHECK: blr
}

define void @a16max(i16* nocapture dereferenceable(4) %minimum, i16 %val) #1 {
entry:
  %0 = atomicrmw max i16* %minimum, i16 %val monotonic
  ret void

; CHECK-LABEL: @a16max
; CHECK: lharx [[OLDV:[0-9]+]], 0, 3
; CHECK: cmpw 4, [[OLDV]]
; CHECK: blelr 0
; CHECK: sthcx. 4, 0, 3
; CHECK: bne 0,
; CHECK: blr
}

define void @a16umin(i16* nocapture dereferenceable(4) %minimum, i16 %val) #1 {
entry:
  %0 = atomicrmw umin i16* %minimum, i16 %val monotonic
  ret void

; CHECK-LABEL: @a16umin
; CHECK: lharx [[OLDV:[0-9]+]], 0, 3
; CHECK: cmplw 4, [[OLDV]]
; CHECK: bgelr 0
; CHECK: sthcx. 4, 0, 3
; CHECK: bne 0,
; CHECK: blr
}

define void @a16umax(i16* nocapture dereferenceable(4) %minimum, i16 %val) #1 {
entry:
  %0 = atomicrmw umax i16* %minimum, i16 %val monotonic
  ret void

; CHECK-LABEL: @a16umax
; CHECK: lharx [[OLDV:[0-9]+]], 0, 3
; CHECK: cmplw 4, [[OLDV]]
; CHECK: blelr 0
; CHECK: sthcx. 4, 0, 3
; CHECK: bne 0,
; CHECK: blr
}

define void @a8min(i8* nocapture dereferenceable(4) %minimum, i8 %val) #1 {
entry:
  %0 = atomicrmw min i8* %minimum, i8 %val monotonic
  ret void

; CHECK-LABEL: @a8min
; CHECK: lbarx [[OLDV:[0-9]+]], 0, 3
; CHECK: cmpw 4, [[OLDV]]
; CHECK: bgelr 0
; CHECK: stbcx. 4, 0, 3
; CHECK: bne 0,
; CHECK: blr
}

define void @a8max(i8* nocapture dereferenceable(4) %minimum, i8 %val) #1 {
entry:
  %0 = atomicrmw max i8* %minimum, i8 %val monotonic
  ret void

; CHECK-LABEL: @a8max
; CHECK: lbarx [[OLDV:[0-9]+]], 0, 3
; CHECK: cmpw 4, [[OLDV]]
; CHECK: blelr 0
; CHECK: stbcx. 4, 0, 3
; CHECK: bne 0,
; CHECK: blr
}

define void @a8umin(i8* nocapture dereferenceable(4) %minimum, i8 %val) #1 {
entry:
  %0 = atomicrmw umin i8* %minimum, i8 %val monotonic
  ret void

; CHECK-LABEL: @a8umin
; CHECK: lbarx [[OLDV:[0-9]+]], 0, 3
; CHECK: cmplw 4, [[OLDV]]
; CHECK: bgelr 0
; CHECK: stbcx. 4, 0, 3
; CHECK: bne 0,
; CHECK: blr
}

define void @a8umax(i8* nocapture dereferenceable(4) %minimum, i8 %val) #1 {
entry:
  %0 = atomicrmw umax i8* %minimum, i8 %val monotonic
  ret void

; CHECK-LABEL: @a8umax
; CHECK: lbarx [[OLDV:[0-9]+]], 0, 3
; CHECK: cmplw 4, [[OLDV]]
; CHECK: blelr 0
; CHECK: stbcx. 4, 0, 3
; CHECK: bne 0,
; CHECK: blr
}

define void @a64min(i64* nocapture dereferenceable(4) %minimum, i64 %val) #0 {
entry:
  %0 = atomicrmw min i64* %minimum, i64 %val monotonic
  ret void

; CHECK-LABEL: @a64min
; CHECK: ldarx [[OLDV:[0-9]+]], 0, 3
; CHECK: cmpd 4, [[OLDV]]
; CHECK: bgelr 0
; CHECK: stdcx. 4, 0, 3
; CHECK: bne 0,
; CHECK: blr
}

define void @a64max(i64* nocapture dereferenceable(4) %minimum, i64 %val) #0 {
entry:
  %0 = atomicrmw max i64* %minimum, i64 %val monotonic
  ret void

; CHECK-LABEL: @a64max
; CHECK: ldarx [[OLDV:[0-9]+]], 0, 3
; CHECK: cmpd 4, [[OLDV]]
; CHECK: blelr 0
; CHECK: stdcx. 4, 0, 3
; CHECK: bne 0,
; CHECK: blr
}

define void @a64umin(i64* nocapture dereferenceable(4) %minimum, i64 %val) #0 {
entry:
  %0 = atomicrmw umin i64* %minimum, i64 %val monotonic
  ret void

; CHECK-LABEL: @a64umin
; CHECK: ldarx [[OLDV:[0-9]+]], 0, 3
; CHECK: cmpld 4, [[OLDV]]
; CHECK: bgelr 0
; CHECK: stdcx. 4, 0, 3
; CHECK: bne 0,
; CHECK: blr
}

define void @a64umax(i64* nocapture dereferenceable(4) %minimum, i64 %val) #0 {
entry:
  %0 = atomicrmw umax i64* %minimum, i64 %val monotonic
  ret void

; CHECK-LABEL: @a64umax
; CHECK: ldarx [[OLDV:[0-9]+]], 0, 3
; CHECK: cmpld 4, [[OLDV]]
; CHECK: blelr 0
; CHECK: stdcx. 4, 0, 3
; CHECK: bne 0,
; CHECK: blr
}

define void @ae16min(i16* nocapture dereferenceable(4) %minimum, i16 %val) #0 {
entry:
  %0 = atomicrmw min i16* %minimum, i16 %val monotonic
  ret void

; CHECK-LABEL: @ae16min
; CHECK-DAG: rlwinm [[SA1:[0-9]+]], 3, 3, 27, 27
; CHECK-DAG: li [[M1:[0-9]+]], 0
; CHECK-DAG: rldicr 3, 3, 0, 61
; CHECK-DAG: xori [[SA:[0-9]+]], [[SA1]], 16
; CHECK-DAG: ori [[M2:[0-9]+]], [[M1]], 65535
; CHECK-DAG: slw [[SV:[0-9]+]], 4, [[SA]]
; CHECK-DAG: slw [[M:[0-9]+]], [[M2]], [[SA]]
; CHECK-DAG: and [[SMV:[0-9]+]], [[SV]], [[M]]
; CHECK: lwarx [[OLDV:[0-9]+]], 0, 3
; CHECK: and [[MOLDV:[0-9]+]], [[OLDV]], [[M]]
; CHECK: srw [[SMOLDV:[0-9]+]], [[MOLDV]], [[SA]]
; CHECK: extsh [[SESMOLDV:[0-9]+]], [[SMOLDV]]
; CHECK: cmpw 0, 4, [[SESMOLDV]]
; CHECK: bgelr 0
; CHECK: andc [[NOLDV:[0-9]+]], [[OLDV]], [[M]]
; CHECK: or [[NEWV:[0-9]+]], [[SMV]], [[NOLDV]]
; CHECK: stwcx. [[NEWV]], 0, 3
; CHECK: bne  0,
; CHECK: blr
}

define void @ae16max(i16* nocapture dereferenceable(4) %minimum, i16 %val) #0 {
entry:
  %0 = atomicrmw max i16* %minimum, i16 %val monotonic
  ret void

; CHECK-LABEL: @ae16max
; CHECK-DAG: rlwinm [[SA1:[0-9]+]], 3, 3, 27, 27
; CHECK-DAG: li [[M1:[0-9]+]], 0
; CHECK-DAG: rldicr 3, 3, 0, 61
; CHECK-DAG: xori [[SA:[0-9]+]], [[SA1]], 16
; CHECK-DAG: ori [[M2:[0-9]+]], [[M1]], 65535
; CHECK-DAG: slw [[SV:[0-9]+]], 4, [[SA]]
; CHECK-DAG: slw [[M:[0-9]+]], [[M2]], [[SA]]
; CHECK-DAG: and [[SMV:[0-9]+]], [[SV]], [[M]]
; CHECK: lwarx [[OLDV:[0-9]+]], 0, 3
; CHECK: and [[MOLDV:[0-9]+]], [[OLDV]], [[M]]
; CHECK: srw [[SMOLDV:[0-9]+]], [[MOLDV]], [[SA]]
; CHECK: extsh [[SESMOLDV:[0-9]+]], [[SMOLDV]]
; CHECK: cmpw 0, 4, [[SESMOLDV]]
; CHECK: blelr 0
; CHECK: andc [[NOLDV:[0-9]+]], [[OLDV]], [[M]]
; CHECK: or [[NEWV:[0-9]+]], [[SMV]], [[NOLDV]]
; CHECK: stwcx. [[NEWV]], 0, 3
; CHECK: bne  0,
; CHECK: blr
}

define void @ae16umin(i16* nocapture dereferenceable(4) %minimum, i16 %val) #0 {
entry:
  %0 = atomicrmw umin i16* %minimum, i16 %val monotonic
  ret void

; CHECK-LABEL: @ae16umin
; CHECK-DAG: rlwinm [[SA1:[0-9]+]], 3, 3, 27, 27
; CHECK-DAG: li [[M1:[0-9]+]], 0
; CHECK-DAG: rldicr 3, 3, 0, 61
; CHECK-DAG: xori [[SA:[0-9]+]], [[SA1]], 16
; CHECK-DAG: ori [[M2:[0-9]+]], [[M1]], 65535
; CHECK-DAG: slw [[SV:[0-9]+]], 4, [[SA]]
; CHECK-DAG: slw [[M:[0-9]+]], [[M2]], [[SA]]
; CHECK-DAG: and [[SMV:[0-9]+]], [[SV]], [[M]]
; CHECK: lwarx [[OLDV:[0-9]+]], 0, 3
; CHECK: and [[MOLDV:[0-9]+]], [[OLDV]], [[M]]
; CHECK: cmplw 0, 4, [[MOLDV]]
; CHECK: bgelr 0
; CHECK: andc [[NOLDV:[0-9]+]], [[OLDV]], [[M]]
; CHECK: or [[NEWV:[0-9]+]], [[SMV]], [[NOLDV]]
; CHECK: stwcx. [[NEWV]], 0, 3
; CHECK: bne  0,
; CHECK: blr
}

define void @ae16umax(i16* nocapture dereferenceable(4) %minimum, i16 %val) #0 {
entry:
  %0 = atomicrmw umax i16* %minimum, i16 %val monotonic
  ret void

; CHECK-LABEL: @ae16umax
; CHECK-DAG: rlwinm [[SA1:[0-9]+]], 3, 3, 27, 27
; CHECK-DAG: li [[M1:[0-9]+]], 0
; CHECK-DAG: rldicr 3, 3, 0, 61
; CHECK-DAG: xori [[SA:[0-9]+]], [[SA1]], 16
; CHECK-DAG: ori [[M2:[0-9]+]], [[M1]], 65535
; CHECK-DAG: slw [[SV:[0-9]+]], 4, [[SA]]
; CHECK-DAG: slw [[M:[0-9]+]], [[M2]], [[SA]]
; CHECK-DAG: and [[SMV:[0-9]+]], [[SV]], [[M]]
; CHECK: lwarx [[OLDV:[0-9]+]], 0, 3
; CHECK: and [[MOLDV:[0-9]+]], [[OLDV]], [[M]]
; CHECK: cmplw 0, 4, [[MOLDV]]
; CHECK: blelr 0
; CHECK: andc [[NOLDV:[0-9]+]], [[OLDV]], [[M]]
; CHECK: or [[NEWV:[0-9]+]], [[SMV]], [[NOLDV]]
; CHECK: stwcx. [[NEWV]], 0, 3
; CHECK: bne  0,
; CHECK: blr
}

define void @ae8min(i8* nocapture dereferenceable(4) %minimum, i8 %val) #0 {
entry:
  %0 = atomicrmw min i8* %minimum, i8 %val monotonic
  ret void

; CHECK-LABEL: @ae8min
; CHECK-DAG: rlwinm [[SA1:[0-9]+]], 3, 3, 27, 28
; CHECK-DAG: li [[M1:[0-9]+]], 255
; CHECK-DAG: rldicr 3, 3, 0, 61
; CHECK-DAG: xori [[SA:[0-9]+]], [[SA1]], 24
; CHECK-DAG: slw [[SV:[0-9]+]], 4, [[SA]]
; CHECK-DAG: slw [[M:[0-9]+]], [[M1]], [[SA]]
; CHECK-DAG: and [[SMV:[0-9]+]], [[SV]], [[M]]
; CHECK: lwarx [[OLDV:[0-9]+]], 0, 3
; CHECK: and [[MOLDV:[0-9]+]], [[OLDV]], [[M]]
; CHECK: srw [[SMOLDV:[0-9]+]], [[MOLDV]], [[SA]]
; CHECK: extsb [[SESMOLDV:[0-9]+]], [[SMOLDV]]
; CHECK: cmpw 0, 4, [[SESMOLDV]]
; CHECK: bgelr 0
; CHECK: andc [[NOLDV:[0-9]+]], [[OLDV]], [[M]]
; CHECK: or [[NEWV:[0-9]+]], [[SMV]], [[NOLDV]]
; CHECK: stwcx. [[NEWV]], 0, 3
; CHECK: bne  0,
; CHECK: blr
}

define void @ae8max(i8* nocapture dereferenceable(4) %minimum, i8 %val) #0 {
entry:
  %0 = atomicrmw max i8* %minimum, i8 %val monotonic
  ret void

; CHECK-LABEL: @ae8max
; CHECK-DAG: rlwinm [[SA1:[0-9]+]], 3, 3, 27, 28
; CHECK-DAG: li [[M1:[0-9]+]], 255
; CHECK-DAG: rldicr 3, 3, 0, 61
; CHECK-DAG: xori [[SA:[0-9]+]], [[SA1]], 24
; CHECK-DAG: slw [[SV:[0-9]+]], 4, [[SA]]
; CHECK-DAG: slw [[M:[0-9]+]], [[M1]], [[SA]]
; CHECK-DAG: and [[SMV:[0-9]+]], [[SV]], [[M]]
; CHECK: lwarx [[OLDV:[0-9]+]], 0, 3
; CHECK: and [[MOLDV:[0-9]+]], [[OLDV]], [[M]]
; CHECK: srw [[SMOLDV:[0-9]+]], [[MOLDV]], [[SA]]
; CHECK: extsb [[SESMOLDV:[0-9]+]], [[SMOLDV]]
; CHECK: cmpw 0, 4, [[SESMOLDV]]
; CHECK: blelr 0
; CHECK: andc [[NOLDV:[0-9]+]], [[OLDV]], [[M]]
; CHECK: or [[NEWV:[0-9]+]], [[SMV]], [[NOLDV]]
; CHECK: stwcx. [[NEWV]], 0, 3
; CHECK: bne  0,
; CHECK: blr
}

define void @ae8umin(i8* nocapture dereferenceable(4) %minimum, i8 %val) #0 {
entry:
  %0 = atomicrmw umin i8* %minimum, i8 %val monotonic
  ret void

; CHECK-LABEL: @ae8umin
; CHECK-DAG: rlwinm [[SA1:[0-9]+]], 3, 3, 27, 28
; CHECK-DAG: li [[M1:[0-9]+]], 255
; CHECK-DAG: rldicr 3, 3, 0, 61
; CHECK-DAG: xori [[SA:[0-9]+]], [[SA1]], 24
; CHECK-DAG: slw [[SV:[0-9]+]], 4, [[SA]]
; CHECK-DAG: slw [[M:[0-9]+]], [[M1]], [[SA]]
; CHECK-DAG: and [[SMV:[0-9]+]], [[SV]], [[M]]
; CHECK: lwarx [[OLDV:[0-9]+]], 0, 3
; CHECK: and [[MOLDV:[0-9]+]], [[OLDV]], [[M]]
; CHECK: cmplw 0, 4, [[MOLDV]]
; CHECK: bgelr 0
; CHECK: andc [[NOLDV:[0-9]+]], [[OLDV]], [[M]]
; CHECK: or [[NEWV:[0-9]+]], [[SMV]], [[NOLDV]]
; CHECK: stwcx. [[NEWV]], 0, 3
; CHECK: bne  0,
; CHECK: blr
}

define void @ae8umax(i8* nocapture dereferenceable(4) %minimum, i8 %val) #0 {
entry:
  %0 = atomicrmw umax i8* %minimum, i8 %val monotonic
  ret void

; CHECK-LABEL: @ae8umax
; CHECK-DAG: rlwinm [[SA1:[0-9]+]], 3, 3, 27, 28
; CHECK-DAG: li [[M1:[0-9]+]], 255
; CHECK-DAG: rldicr 3, 3, 0, 61
; CHECK-DAG: xori [[SA:[0-9]+]], [[SA1]], 24
; CHECK-DAG: slw [[SV:[0-9]+]], 4, [[SA]]
; CHECK-DAG: slw [[M:[0-9]+]], [[M1]], [[SA]]
; CHECK-DAG: and [[SMV:[0-9]+]], [[SV]], [[M]]
; CHECK: lwarx [[OLDV:[0-9]+]], 0, 3
; CHECK: and [[MOLDV:[0-9]+]], [[OLDV]], [[M]]
; CHECK: cmplw 0, 4, [[MOLDV]]
; CHECK: blelr 0
; CHECK: andc [[NOLDV:[0-9]+]], [[OLDV]], [[M]]
; CHECK: or [[NEWV:[0-9]+]], [[SMV]], [[NOLDV]]
; CHECK: stwcx. [[NEWV]], 0, 3
; CHECK: bne  0,
; CHECK: blr
}

attributes #0 = { nounwind "target-cpu"="ppc64" }
attributes #1 = { nounwind "target-cpu"="pwr8" }

