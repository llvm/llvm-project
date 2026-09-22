; RUN: opt -S -passes=slp-vectorizer -mtriple=aarch64 -slp-threshold=-1 -slp-use-vplan-codegen < %s | FileCheck %s

declare void @may_write()
declare void @read_only() memory(read)

; The loads must not be sunk past a call that may write memory.
define void @call_between_loads_and_root(ptr %p, ptr %q, ptr %r) {
; CHECK-LABEL: @call_between_loads_and_root(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = load <2 x double>, ptr [[P:%.*]], align 8
; CHECK-NEXT:    [[TMP1:%.*]] = load <2 x double>, ptr [[Q:%.*]], align 8
; CHECK-NEXT:    [[TMP2:%.*]] = fadd <2 x double> [[TMP0]], [[TMP1]]
; CHECK-NEXT:    call void @may_write()
; CHECK-NEXT:    store <2 x double> [[TMP2]], ptr [[R:%.*]], align 8
; CHECK-NEXT:    ret void
;
entry:
  %a0 = load double, ptr %p, align 8
  %p1 = getelementptr inbounds double, ptr %p, i64 1
  %a1 = load double, ptr %p1, align 8
  %b0 = load double, ptr %q, align 8
  %q1 = getelementptr inbounds double, ptr %q, i64 1
  %b1 = load double, ptr %q1, align 8
  %s0 = fadd double %a0, %b0
  %s1 = fadd double %a1, %b1
  call void @may_write()
  store double %s0, ptr %r, align 8
  %r1 = getelementptr inbounds double, ptr %r, i64 1
  store double %s1, ptr %r1, align 8
  ret void
}

; Same for a store that may alias the loads.
define void @store_between_loads_and_root(ptr %p, ptr %q, ptr %r, ptr %x) {
; CHECK-LABEL: @store_between_loads_and_root(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = load <2 x double>, ptr [[P:%.*]], align 8
; CHECK-NEXT:    [[TMP1:%.*]] = load <2 x double>, ptr [[Q:%.*]], align 8
; CHECK-NEXT:    [[TMP2:%.*]] = fadd <2 x double> [[TMP0]], [[TMP1]]
; CHECK-NEXT:    store i8 0, ptr [[X:%.*]], align 1
; CHECK-NEXT:    store <2 x double> [[TMP2]], ptr [[R:%.*]], align 8
; CHECK-NEXT:    ret void
;
entry:
  %a0 = load double, ptr %p, align 8
  %p1 = getelementptr inbounds double, ptr %p, i64 1
  %a1 = load double, ptr %p1, align 8
  %b0 = load double, ptr %q, align 8
  %q1 = getelementptr inbounds double, ptr %q, i64 1
  %b1 = load double, ptr %q1, align 8
  %s0 = fadd double %a0, %b0
  %s1 = fadd double %a1, %b1
  store i8 0, ptr %x, align 1
  store double %s0, ptr %r, align 8
  %r1 = getelementptr inbounds double, ptr %r, i64 1
  store double %s1, ptr %r1, align 8
  ret void
}

; A read-only call can be crossed.
define void @read_only_call_between_loads_and_root(ptr %p, ptr %q, ptr %r) {
; CHECK-LABEL: @read_only_call_between_loads_and_root(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @read_only()
; CHECK-NEXT:    %wide.load = load <2 x double>, ptr %p, align 8
; CHECK-NEXT:    %wide.load1 = load <2 x double>, ptr %q, align 8
; CHECK-NEXT:    [[TMP0:%.*]] = fadd <2 x double> %wide.load, %wide.load1
; CHECK-NEXT:    store <2 x double> [[TMP0]], ptr %r, align 8
; CHECK-NEXT:    ret void
;
entry:
  %a0 = load double, ptr %p, align 8
  %p1 = getelementptr inbounds double, ptr %p, i64 1
  %a1 = load double, ptr %p1, align 8
  %b0 = load double, ptr %q, align 8
  %q1 = getelementptr inbounds double, ptr %q, i64 1
  %b1 = load double, ptr %q1, align 8
  %s0 = fadd double %a0, %b0
  %s1 = fadd double %a1, %b1
  call void @read_only()
  store double %s0, ptr %r, align 8
  %r1 = getelementptr inbounds double, ptr %r, i64 1
  store double %s1, ptr %r1, align 8
  ret void
}

; A call after the root is not crossed.
define void @call_after_root(ptr %p, ptr %q, ptr %r) {
; CHECK-LABEL: @call_after_root(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %wide.load = load <2 x double>, ptr %p, align 8
; CHECK-NEXT:    %wide.load1 = load <2 x double>, ptr %q, align 8
; CHECK-NEXT:    [[TMP0:%.*]] = fadd <2 x double> %wide.load, %wide.load1
; CHECK-NEXT:    store <2 x double> [[TMP0]], ptr %r, align 8
; CHECK-NEXT:    call void @may_write()
; CHECK-NEXT:    ret void
;
entry:
  %a0 = load double, ptr %p, align 8
  %p1 = getelementptr inbounds double, ptr %p, i64 1
  %a1 = load double, ptr %p1, align 8
  %b0 = load double, ptr %q, align 8
  %q1 = getelementptr inbounds double, ptr %q, i64 1
  %b1 = load double, ptr %q1, align 8
  %s0 = fadd double %a0, %b0
  %s1 = fadd double %a1, %b1
  store double %s0, ptr %r, align 8
  %r1 = getelementptr inbounds double, ptr %r, i64 1
  store double %s1, ptr %r1, align 8
  call void @may_write()
  ret void
}

; Scheduling moves the interleaved root stores after the loads.
define void @interleaved_root_stores(ptr noalias %p, ptr noalias %r) {
; CHECK-LABEL: @interleaved_root_stores(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %wide.load = load <2 x double>, ptr %p, align 8
; CHECK-NEXT:    [[TMP0:%.*]] = fneg <2 x double> %wide.load
; CHECK-NEXT:    store <2 x double> [[TMP0]], ptr %r, align 8
; CHECK-NEXT:    ret void
;
entry:
  %a0 = load double, ptr %p, align 8
  %n0 = fneg double %a0
  store double %n0, ptr %r, align 8
  %p1 = getelementptr inbounds double, ptr %p, i64 1
  %a1 = load double, ptr %p1, align 8
  %n1 = fneg double %a1
  %r1 = getelementptr inbounds double, ptr %r, i64 1
  store double %n1, ptr %r1, align 8
  ret void
}
