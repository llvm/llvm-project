; For ELFv2 ABI, we can avoid allocating the parameter area in the stack frame of the caller function
; if all the arguments can be passed to the callee in registers.
; For ELFv1 ABI, we always need to allocate the parameter area.

; Tests for ELFv2 ABI
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -target-abi elfv2 < %s | FileCheck %s -check-prefix=PPC64-ELFV2
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -target-abi elfv2 < %s | FileCheck %s -check-prefix=PPC64-ELFV2

; Tests for ELFv1 ABI
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -target-abi elfv1 < %s | FileCheck %s -check-prefix=PPC64-ELFV1
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -target-abi elfv1 < %s | FileCheck %s -check-prefix=PPC64-ELFV1

; If the callee has at most eight integer args, parameter area can be ommited for ELFv2 ABI.

; PPC64-ELFV2-LABEL: WithoutParamArea1:
; PPC64-ELFV2-NOT: stw {{[0-9]+}}, -{{[0-9]+}}(1)
; PPC64-ELFV2: stdu 1, -32(1)
; PPC64-ELFV2: addi 1, 1, 32
; PPC64-ELFV2-NOT: lwz {{[0-9]+}}, -{{[0-9]+}}(1)
; PPC64-ELFV1-LABEL: WithoutParamArea1:
; PPC64-ELFV1-NOT: stw {{[0-9]+}}, -{{[0-9]+}}(1)
; PPC64-ELFV1: stdu 1, -112(1)
; PPC64-ELFV1: addi 1, 1, 112
; PPC64-ELFV1-NOT: lwz {{[0-9]+}}, -{{[0-9]+}}(1)
define signext i32 @WithoutParamArea1(i32 signext %a) local_unnamed_addr #0 {
entry:
  %call = tail call signext i32 @onearg(i32 signext %a) #2
  ret i32 %call
}

; PPC64-ELFV2-LABEL: WithoutParamArea2:
; PPC64-ELFV2-NOT: stw {{[0-9]+}}, -{{[0-9]+}}(1)
; PPC64-ELFV2: stdu 1, -32(1)
; PPC64-ELFV2: addi 1, 1, 32
; PPC64-ELFV2-NOT: lwz {{[0-9]+}}, -{{[0-9]+}}(1)
; PPC64-ELFV1-LABEL: WithoutParamArea2:
; PPC64-ELFV1-NOT: stw {{[0-9]+}}, -{{[0-9]+}}(1)
; PPC64-ELFV1: stdu 1, -112(1)
; PPC64-ELFV1: addi 1, 1, 112
; PPC64-ELFV1-NOT: lwz {{[0-9]+}}, -{{[0-9]+}}(1)
define signext i32 @WithoutParamArea2(i32 signext %a) local_unnamed_addr #0 {
entry:
  %call = tail call signext i32 @eightargs(i32 signext %a, i32 signext %a, i32 signext %a, i32 signext %a, i32 signext %a, i32 signext %a, i32 signext %a, i32 signext %a) #2
  ret i32 %call
}

; If the callee has more than eight integer args or variable number of args, 
; parameter area cannot be ommited even for ELFv2 ABI

; PPC64-ELFV2-LABEL: WithParamArea1:
; PPC64-ELFV2-NOT: stw {{[0-9]+}}, -{{[0-9]+}}(1)
; PPC64-ELFV2: stdu 1, -96(1)
; PPC64-ELFV2: addi 1, 1, 96
; PPC64-ELFV2-NOT: lwz {{[0-9]+}}, -{{[0-9]+}}(1)
; PPC64-ELFV1-LABEL: WithParamArea1:
; PPC64-ELFV1-NOT: stw {{[0-9]+}}, -{{[0-9]+}}(1)
; PPC64-ELFV1: stdu 1, -112(1)
; PPC64-ELFV1: addi 1, 1, 112
; PPC64-ELFV1-NOT: lwz {{[0-9]+}}, -{{[0-9]+}}(1)
define signext i32 @WithParamArea1(i32 signext %a) local_unnamed_addr #0 {
entry:
  %call = tail call signext i32 (i32, ...) @varargs(i32 signext %a, i32 signext %a) #2
  ret i32 %call
}

; PPC64-ELFV2-LABEL: WithParamArea2:
; PPC64-ELFV2-NOT: stw {{[0-9]+}}, -{{[0-9]+}}(1)
; PPC64-ELFV2: stdu 1, -112(1)
; PPC64-ELFV2: addi 1, 1, 112
; PPC64-ELFV2-NOT: lwz {{[0-9]+}}, -{{[0-9]+}}(1)
; PPC64-ELFV1-LABEL: WithParamArea2:
; PPC64-ELFV1-NOT: stw {{[0-9]+}}, -{{[0-9]+}}(1)
; PPC64-ELFV1: stdu 1, -128(1)
; PPC64-ELFV1: addi 1, 1, 128
; PPC64-ELFV1-NOT: lwz {{[0-9]+}}, -{{[0-9]+}}(1)
define signext i32 @WithParamArea2(i32 signext %a) local_unnamed_addr #0 {
entry:
  %call = tail call signext i32 @nineargs(i32 signext %a, i32 signext %a, i32 signext %a, i32 signext %a, i32 signext %a, i32 signext %a, i32 signext %a, i32 signext %a, i32 signext %a) #2
  ret i32 %call
}

declare signext i32 @onearg(i32 signext) local_unnamed_addr #1
declare signext i32 @eightargs(i32 signext, i32 signext, i32 signext, i32 signext, i32 signext, i32 signext, i32 signext, i32 signext) local_unnamed_addr #1
declare signext i32 @nineargs(i32 signext, i32 signext, i32 signext, i32 signext, i32 signext, i32 signext, i32 signext, i32 signext, i32 signext) local_unnamed_addr #1
declare signext i32 @varargs(i32 signext, ...) local_unnamed_addr #1

