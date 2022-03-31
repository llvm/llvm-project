; RUN: llc < %s -mtriple armv8a--none-eabi -mattr=+fullfp16 | FileCheck %s

define void @load_zero(half* %in, half* %out) {
entry:
; CHECK-LABEL: load_zero:
; CHECK: vldr.16 {{s[0-9]+}}, [r0]
  %arrayidx = getelementptr inbounds half, half* %in, i32 0
  %load = load half, half* %arrayidx, align 2
  store half %load, half* %out
  ret void
}

define void @load_255(half* %in, half* %out) {
entry:
; CHECK-LABEL: load_255:
; CHECK: vldr.16 {{s[0-9]+}}, [r0, #510]
  %arrayidx = getelementptr inbounds half, half* %in, i32 255
  %load = load half, half* %arrayidx, align 2
  store half %load, half* %out
  ret void
}

define void @load_256(half* %in, half* %out) {
entry:
; CHECK-LABEL: load_256:
; CHECK: add     [[ADDR:r[0-9]+]], r0, #512
; CHECK: vldr.16 {{s[0-9]+}}, [[[ADDR]]]
  %arrayidx = getelementptr inbounds half, half* %in, i32 256
  %load = load half, half* %arrayidx, align 2
  store half %load, half* %out
  ret void
}

define void @load_neg_255(half* %in, half* %out) {
entry:
; CHECK-LABEL: load_neg_255:
; CHECK: vldr.16 {{s[0-9]+}}, [r0, #-510]
  %arrayidx = getelementptr inbounds half, half* %in, i32 -255
  %load = load half, half* %arrayidx, align 2
  store half %load, half* %out
  ret void
}

define void @load_neg_256(half* %in, half* %out) {
entry:
; CHECK-LABEL: load_neg_256:
; CHECK: sub     [[ADDR:r[0-9]+]], r0, #512
; CHECK: vldr.16 {{s[0-9]+}}, [[[ADDR]]]
  %arrayidx = getelementptr inbounds half, half* %in, i32 -256
  %load = load half, half* %arrayidx, align 2
  store half %load, half* %out
  ret void
}

define void @store_zero(half* %in, half* %out) {
entry:
; CHECK-LABEL: store_zero:
  %load = load half, half* %in, align 2
; CHECK: vstr.16 {{s[0-9]+}}, [r1]
  %arrayidx = getelementptr inbounds half, half* %out, i32 0
  store half %load, half* %arrayidx
  ret void
}

define void @store_255(half* %in, half* %out) {
entry:
; CHECK-LABEL: store_255:
  %load = load half, half* %in, align 2
; CHECK: vstr.16 {{s[0-9]+}}, [r1, #510]
  %arrayidx = getelementptr inbounds half, half* %out, i32 255
  store half %load, half* %arrayidx
  ret void
}

define void @store_256(half* %in, half* %out) {
entry:
; CHECK-LABEL: store_256:
  %load = load half, half* %in, align 2
; CHECK: add     [[ADDR:r[0-9]+]], r1, #512
; CHECK: vstr.16 {{s[0-9]+}}, [[[ADDR]]]
  %arrayidx = getelementptr inbounds half, half* %out, i32 256
  store half %load, half* %arrayidx
  ret void
}

define void @store_neg_255(half* %in, half* %out) {
entry:
; CHECK-LABEL: store_neg_255:
  %load = load half, half* %in, align 2
; CHECK: vstr.16 {{s[0-9]+}}, [r1, #-510]
  %arrayidx = getelementptr inbounds half, half* %out, i32 -255
  store half %load, half* %arrayidx
  ret void
}

define void @store_neg_256(half* %in, half* %out) {
entry:
; CHECK-LABEL: store_neg_256:
  %load = load half, half* %in, align 2
; CHECK: sub     [[ADDR:r[0-9]+]], r1, #512
; CHECK: vstr.16 {{s[0-9]+}}, [[[ADDR]]]
  %arrayidx = getelementptr inbounds half, half* %out, i32 -256
  store half %load, half* %arrayidx
  ret void
}
