; REQUIRES: amdgpu-registered-target

; RUN: llvm-mc %s --triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj --amdhsa-code-object-version=6 --amdgpu-force-generic-version=1 -o %t.o
; RUN: llvm-readelf -h %t.o   | FileCheck %s --check-prefix=V1

; RUN: llvm-mc %s --triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj --amdhsa-code-object-version=6 --amdgpu-force-generic-version=4 -o %t.o
; RUN: llvm-readelf -h %t.o   | FileCheck %s --check-prefix=V4

; RUN: llvm-mc %s --triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj --amdhsa-code-object-version=6 --amdgpu-force-generic-version=32 -o %t.o
; RUN: llvm-readelf -h %t.o   | FileCheck %s --check-prefix=V32

; RUN: llvm-mc %s --triple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj --amdhsa-code-object-version=6 --amdgpu-force-generic-version=255 -o %t.o
; RUN: llvm-readelf -h %t.o   | FileCheck %s --check-prefix=V255

; V1: generic_v1
; V4: generic_v4
; V32: generic_v32
; V255: generic_v255
