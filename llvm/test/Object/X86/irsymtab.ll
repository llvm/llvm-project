; RUN: env LLVM_OVERRIDE_PRODUCER=producer opt -o %t %s
; RUN: llvm-bcanalyzer -dump -show-binary-blobs %t | FileCheck --check-prefix=BCA %s

; Same producer, does not require upgrade.
; RUN: env LLVM_OVERRIDE_PRODUCER=producer llvm-lto2 dump-symtab %t | FileCheck --check-prefix=SYMTAB %s

; Different producer, requires upgrade.
; RUN: env LLVM_OVERRIDE_PRODUCER=consumer llvm-lto2 dump-symtab %t | FileCheck --check-prefix=SYMTAB %s

; BCA:      <SYMTAB_BLOCK
; Version stored at offset 0.
; BCA-NEXT:   <BLOB abbrevid=4/> blob data = '\x01\x00\x00\x00\x06\x00\x00\x00\x08\x00\x00\x00D\x00\x00\x00\x01\x00\x00\x00P\x00\x00\x00\x00\x00\x00\x00P\x00\x00\x00\x02\x00\x00\x00\x80\x00\x00\x00\x00\x00\x00\x00\x0E\x00\x00\x00\x18\x00\x00\x00&\x00\x00\x00\x0B\x00\x00\x001\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\xFF\xFF\xFF\xFF\x00$\x00\x00\x03\x00\x00\x00\x03\x00\x00\x00\x03\x00\x00\x00\x03\x00\x00\x00\xFF\xFF\xFF\xFF\x08$\x00\x00'
; BCA-NEXT: </SYMTAB_BLOCK>
; BCA-NEXT: <STRTAB_BLOCK
; BCA-NEXT:   <BLOB abbrevid=4/> blob data = 'foobarproducerx86_64-unknown-linux-gnuirsymtab.ll'
; BCA-NEXT: </STRTAB_BLOCK>

; SYMTAB:      version: 1
; SYMTAB-NEXT: producer: producer
; SYMTAB-NEXT: target triple: x86_64-unknown-linux-gnu
; SYMTAB-NEXT: source filename: irsymtab.ll
; SYMTAB-NEXT: D------X foo
; SYMTAB-NEXT: DU-----X bar

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
source_filename = "irsymtab.ll"

define void @foo() {
  ret void
}

declare void @bar()
