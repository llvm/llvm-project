; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN: -code-model=small < %s | FileCheck %s --check-prefix=SMALL

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff \
; RUN: -code-model=large < %s | FileCheck %s --check-prefix=LARGE

@a = common global i32 0

define zeroext i32 @test_load() {
entry:
  %0 = load i32, i32* @a
  ret i32 %0
}

; SMALL-LABEL: .test_load:{{$}}
; SMALL: ld [[REG1:[0-9]+]], L..C0(2)
; SMALL: lwz [[REG2:[0-9]+]], 0([[REG1]])
; SMALL: blr

; LARGE-LABEL: .test_load:{{$}}
; LARGE: addis [[REG1:[0-9]+]], L..C0@u(2)
; LARGE: ld [[REG2:[0-9]+]], L..C0@l([[REG1]])
; LARGE: lwz [[REG3:[0-9]+]], 0([[REG2]])
; LARGE: blr

@b = common global i32 0

define void @test_store(i32 zeroext %0) {
  store i32 %0, i32* @b
  ret void
}

; SMALL-LABEL: .test_store:{{$}}
; SMALL: ld [[REG1:[0-9]+]], L..C1(2)
; SMALL: stw [[REG2:[0-9]+]], 0([[REG1]])
; SMALL: blr

; LARGE-LABEL: .test_store:{{$}}
; LARGE: addis [[REG1:[0-9]+]], L..C1@u(2)
; LARGE: ld [[REG2:[0-9]+]], L..C1@l([[REG1]])
; LARGE: stw [[REG3:[0-9]+]], 0([[REG2]])
; LARGE: blr

; SMALL: .tc a[TC],a
; SMALL: .tc b[TC],b

; LARGE: .tc a[TE],a
; LARGE: .tc b[TE],b
