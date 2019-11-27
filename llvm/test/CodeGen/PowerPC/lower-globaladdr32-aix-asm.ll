; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix-xcoff \
; RUN: -code-model=small < %s | FileCheck %s --check-prefixes=CHECK,SMALL

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mtriple powerpc-ibm-aix-xcoff \
; RUN: -code-model=large < %s | FileCheck %s --check-prefixes=CHECK,LARGE

@a = common global i32 0

define i32 @test_load() {
entry:
  %0 = load i32, i32* @a
  ret i32 %0
}

; SMALL-LABEL: .test_load:{{$}}
; SMALL: lwz [[REG1:[0-9]+]], LC0(2)
; SMALL: lwz [[REG2:[0-9]+]], 0([[REG1]])
; SMALL: blr

; LARGE-LABEL: .test_load:{{$}}
; LARGE: addis [[REG1:[0-9]+]], LC0@u(2)
; LARGE: lwz [[REG2:[0-9]+]], LC0@l([[REG1]])
; LARGE: lwz [[REG3:[0-9]+]], 0([[REG2]])
; LARGE: blr

@b = common global i32 0

define void @test_store(i32 %0) {
  store i32 %0, i32* @b
  ret void
}

; SMALL-LABEL: .test_store:{{$}}
; SMALL: lwz [[REG1:[0-9]+]], LC1(2)
; SMALL: stw [[REG2:[0-9]+]], 0([[REG1]])
; SMALL: blr

; LARGE-LABEL: .test_store:{{$}}
; LARGE: addis [[REG1:[0-9]+]], LC1@u(2)
; LARGE: lwz [[REG2:[0-9]+]], LC1@l([[REG1]])
; LARGE: stw [[REG3:[0-9]+]], 0([[REG2]])
; LARGE: blr

; CHECK: .tc a[TC],a
; CHECK: .tc b[TC],b
