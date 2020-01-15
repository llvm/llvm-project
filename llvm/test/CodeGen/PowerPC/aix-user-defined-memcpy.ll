; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN: -mattr=-altivec -filetype=obj -o %t.o < %s

; RUN: llvm-readobj --syms %t.o | FileCheck --check-prefix=32-SYM %s

; RUN: llvm-readobj --relocs --expand-relocs %t.o | FileCheck \
; RUN: --check-prefix=32-REL %s

; RUN: not llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff \
; RUN: -mcpu=pwr4 -mattr=-altivec -filetype=obj < %s 2>&1 | FileCheck \
; RUN: --check-prefix=64-CHECK %s

; Test verifies:
; If there exists a user-defined function whose name is the same as the
; "memcpy" ExternalSymbol's, we pick up the user-defined version, even if this
; may lead to some undefined behavior.

define dso_local signext i32 @memcpy(i8* %destination, i32 signext %num) {
entry:
  ret i32 3
}

define void @call_memcpy(i8* %p, i8* %q, i32 %n) {
entry:
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %p, i8* %q, i32 %n, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1)

; TODO: This test should preferably check the symbol table for .o file and
;       the relocation associated with the call.

; 32-SYM:      Symbol {{[{][[:space:]] *}}Index: [[#Index:]]{{[[:space:]] *}}Name: .memcpy 
; 32-SYM-NEXT:    Value (RelocatableAddress): 0x0
; 32-SYM-NEXT:    Section: .text
; 32-SYM-NEXT:    Type: 0x0
; 32-SYM-NEXT:    StorageClass: C_EXT (0x2)
; 32-SYM-NEXT:    NumberOfAuxEntries: 1
; 32-SYM-NEXT:    CSECT Auxiliary Entry {
; 32-SYM-NEXT:      Index: 3
; 32-SYM-NEXT:      ContainingCsectSymbolIndex: 0
; 32-SYM-NEXT:      ParameterHashIndex: 0x0
; 32-SYM-NEXT:      TypeChkSectNum: 0x0
; 32-SYM-NEXT:      SymbolAlignmentLog2: 0
; 32-SYM-NEXT:      SymbolType: XTY_LD (0x2)
; 32-SYM-NEXT:      StorageMappingClass: XMC_PR (0x0)
; 32-SYM-NEXT:      StabInfoIndex: 0x0
; 32-SYM-NEXT:      StabSectNum: 0x0
; 32-SYM-NEXT:    }
; 32-SYM-NEXT:  }

; 32-SYM-NOT: .memcpy

; We are expecting to have the test fail when the support for relocations land.
; 32-REL-NOT: Relocation{{[[:space:]]}}

; 64-CHECK: LLVM ERROR: 64-bit XCOFF object files are not supported yet.
