; Test file to verify the prologue and epilogue insertion point computation by the shrink-wrap pass

; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-unknown -mcpu=pwr9 | FileCheck %s --check-prefixes=POWERPC64
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-ibm-aix-xcoff -mcpu=pwr9 | FileCheck %s --check-prefixes=POWERPC32-AIX
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-ibm-aix-xcoff -mcpu=pwr9 | FileCheck %s --check-prefixes=POWERPC64-AIX

define void @foo(ptr noundef readnone %parent_frame_pointer) {
; POWERPC64-LABEL:      foo
; POWERPC64:            # %bb.0:
; POWERPC64-NEXT:           cmpld [[REG1:[0-9]+]], 1
; POWERPC64:            # %bb.1:
; POWERPC64-NEXT:           mflr [[REG2:[0-9]+]]
; POWERPC64-NEXT:           stdu 1, -32(1)

; POWERPC32-AIX-LABEL:  .foo:
; POWERPC32-AIX:        # %bb.0:
; POWERPC32-AIX-NEXT:       cmplw [[REG1:[0-9]+]], 1
; POWERPC32-AIX:        # %bb.1:
; POWERPC32-AIX-NEXT:       mflr [[REG2:[0-9]+]]
; POWERPC32-AIX-NEXT:       stwu 1, -64(1)

; POWERPC64-AIX-LABEL:  .foo:
; POWERPC64-AIX:        # %bb.0:
; POWERPC64-AIX-NEXT:       cmpld [[REG1:[0-9]+]], 1
; POWERPC64-AIX:        # %bb.1:
; POWERPC64-AIX-NEXT:       mflr [[REG2:[0-9]+]]
; POWERPC64-AIX-NEXT:       stdu 1, -112(1)

entry:
  %frameaddress = tail call ptr @llvm.frameaddress.p0(i32 0)
  %cmp = icmp ugt ptr %parent_frame_pointer, %frameaddress
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  tail call void @abort()
  unreachable

cond.end:                                         ; preds = %entry
  ret void
}

declare ptr @llvm.frameaddress.p0(i32 immarg)
declare void @abort()

define noundef i32 @main() {
; POWERPC64-LABEL:      main
; POWERPC64:            # %bb.0:
; POWERPC64-NEXT:           mflr [[REG1:[0-9]+]]
; POWERPC64-NEXT:           stdu 1, -32(1)
; POWERPC64-NEXT:           std [[REG1]], 48(1)
; POWERPC64:                mr [[REG2:[0-9]+]], 1
; POWERPC64:                addi 1, 1, 32
; POWERPC64-NEXT:           ld [[REG1]], 16(1)
; POWERPC64-NEXT:           mtlr [[REG1]]
; POWERPC64-NEXT:           blr

; POWERPC32-AIX-LABEL:  .main:
; POWERPC32-AIX:        # %bb.0:
; POWERPC32-AIX-NEXT:       mflr [[REG1:[0-9]+]]
; POWERPC32-AIX-NEXT:       stwu 1, -64(1)
; POWERPC32-AIX-NEXT:       mr [[REG2:[0-9]+]], 1
; POWERPC32-AIX-NEXT:       stw [[REG1]], 72(1)
; POWERPC32-AIX:            addi 1, 1, 64
; POWERPC32-AIX-NEXT:       lwz [[REG1]], 8(1)
; POWERPC32-AIX-NEXT:       mtlr [[REG1]]
; POWERPC32-AIX-NEXT:       blr

; POWERPC64-AIX-LABEL: .main:
; POWERPC64-AIX:       # %bb.0:
; POWERPC64-AIX-NEXT:       mflr [[REG1:[0-9]+]]
; POWERPC64-AIX-NEXT:       stdu 1, -112(1)
; POWERPC64-AIX-NEXT:       mr [[REG2:[0-9]+]], 1
; POWERPC64-AIX-NEXT:       std [[REG1]], 128(1)
; POWERPC64-AIX:            addi 1, 1, 112
; POWERPC64-AIX-NEXT:       ld [[REG1]], 16(1)
; POWERPC64-AIX-NEXT:       mtlr [[REG1]]
; POWERPC64-AIX-NEXT:       blr

entry:
  %frameaddress = tail call ptr @llvm.frameaddress.p0(i32 0)
  tail call void @foo(ptr noundef %frameaddress)
  ret i32 0
}
