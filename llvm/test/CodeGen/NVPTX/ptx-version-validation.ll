; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_103a -mattr=+ptx90 2>&1 | FileCheck %s --check-prefix=CHECK-SM103A-HIGH
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_103a 2>&1 | FileCheck %s --check-prefix=CHECK-SM103A
; RUN: not llc < %s -mtriple=nvptx64 -mcpu=sm_103a -mattr=+ptx87 2>&1 | FileCheck %s --check-prefix=CHECK-SM103A-LOW
; RUN: %if ptxas-sm_103a && ptxas-isa-9.0 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_103a -mattr=+ptx90 | %ptxas-verify -arch=sm_103a %}
; RUN: %if ptxas-sm_103a %{ llc < %s -mtriple=nvptx64 -mcpu=sm_103a | %ptxas-verify -arch=sm_103a %}

; Test that sm_120a defaults/requires PTX 8.7
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_120a 2>&1 | FileCheck %s --check-prefix=CHECK-SM120A
; RUN: not llc < %s -mtriple=nvptx64 -mcpu=sm_120a -mattr=+ptx86 2>&1 | FileCheck %s --check-prefix=CHECK-SM120A-LOW
; RUN: %if ptxas-sm_120a %{ llc < %s -mtriple=nvptx64 -mcpu=sm_120a | %ptxas-verify -arch=sm_120a %}

; Test that sm_90a defaults/requires PTX 8.0
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90a 2>&1 | FileCheck %s --check-prefix=CHECK-SM90A
; RUN: not llc < %s -mtriple=nvptx64 -mcpu=sm_90a -mattr=+ptx78 2>&1 | FileCheck %s --check-prefix=CHECK-SM90A-LOW
; RUN: %if ptxas-sm_90a %{ llc < %s -mtriple=nvptx64 -mcpu=sm_90a | %ptxas-verify -arch=sm_90a %}

; Test older SM targets
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 2>&1 | FileCheck %s --check-prefix=CHECK-SM80
; RUN: not llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx63 2>&1 | FileCheck %s --check-prefix=CHECK-SM80-LOW
; RUN: %if ptxas-sm_80 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_80 | %ptxas-verify -arch=sm_80 %}

; CHECK-SM103A-HIGH: .version 9.0
; CHECK-SM103A-HIGH: .target sm_103a

; CHECK-SM103A: .version 8.8
; CHECK-SM103A: .target sm_103a

; CHECK-SM103A-LOW: LLVM ERROR: PTX version 8.7 does not support target 'sm_103a'.
; CHECK-SM103A-LOW: Minimum required PTX version is 8.8.

; CHECK-SM120A: .version 8.7
; CHECK-SM120A: .target sm_120a

; CHECK-SM120A-LOW: LLVM ERROR: PTX version 8.6 does not support target 'sm_120a'.
; CHECK-SM120A-LOW: Minimum required PTX version is 8.7.

; CHECK-SM90A: .version 8.0
; CHECK-SM90A: .target sm_90a

; CHECK-SM90A-LOW: LLVM ERROR: PTX version 7.8 does not support target 'sm_90a'.
; CHECK-SM90A-LOW: Minimum required PTX version is 8.0.

; CHECK-SM80: .version 7.0
; CHECK-SM80: .target sm_80

; CHECK-SM80-LOW: LLVM ERROR: PTX version 6.3 does not support target 'sm_80'.
; CHECK-SM80-LOW: Minimum required PTX version is 7.0.

define void @foo() {
  ret void
}
