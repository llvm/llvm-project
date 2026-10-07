; PTX ISA 9.0 renamed sm_101, sm_101f and sm_101a to sm_110, sm_110f and
; sm_110a. Naming one of these architectures with -mcpu emits `.target` with the
; spelling the PTX version in use knows, so that the sm_110 names work on the PTX
; versions predating the rename.

; RUN: llc < %s -mcpu=sm_110 | FileCheck %s --check-prefix=SM101
; RUN: llc < %s -mcpu=sm_110 -mattr=+ptx88 | FileCheck %s --check-prefix=SM101-PTX88
; RUN: llc < %s -mcpu=sm_110 -mattr=+ptx90 | FileCheck %s --check-prefix=SM110

; RUN: llc < %s -mcpu=sm_110f | FileCheck %s --check-prefix=SM101F
; RUN: llc < %s -mcpu=sm_110f -mattr=+ptx90 | FileCheck %s --check-prefix=SM110F

; RUN: llc < %s -mcpu=sm_110a | FileCheck %s --check-prefix=SM101A
; RUN: llc < %s -mcpu=sm_110a -mattr=+ptx90 | FileCheck %s --check-prefix=SM110A

; RUN: %if ptxas-sm_101 && ptxas-isa-8.6 %{ llc < %s -mcpu=sm_110 | %ptxas-verify -arch=sm_101 %}
; RUN: %if ptxas-sm_110 && ptxas-isa-9.0 %{ llc < %s -mcpu=sm_110 -mattr=+ptx90 | %ptxas-verify -arch=sm_110 %}

; SM101: .version 8.6
; SM101: .target sm_101

; SM101-PTX88: .version 8.8
; SM101-PTX88: .target sm_101

; SM110: .version 9.0
; SM110: .target sm_110

; SM101F: .version 8.8
; SM101F: .target sm_101f

; SM110F: .version 9.0
; SM110F: .target sm_110f

; SM101A: .version 8.6
; SM101A: .target sm_101a

; SM110A: .version 9.0
; SM110A: .target sm_110a

target triple = "nvptx64-nvidia-cuda"

define void @foo() {
  ret void
}
