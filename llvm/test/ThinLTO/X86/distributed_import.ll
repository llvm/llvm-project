; Test distributed build thin link output from llvm-lto2

; Generate bitcode files with summary, as well as minimized bitcode without
; the debug metadata for the thin link.
; RUN: opt -thinlto-bc %s -thin-link-bitcode-file=%t1.thinlink.bc -o %t1.bc
; RUN: opt -thinlto-bc %p/Inputs/distributed_import.ll -thin-link-bitcode-file=%t2.thinlink.bc -o %t2.bc
; RUN: llvm-bcanalyzer -dump %t1.thinlink.bc | FileCheck --check-prefix=THINLINKBITCODE %s

; First perform the thin link on the normal bitcode file.
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -thinlto-distributed-indexes \
; RUN:     -r=%t1.bc,g, \
; RUN:     -r=%t1.bc,analias, \
; RUN:     -r=%t1.bc,f,px \
; RUN:     -r=%t2.bc,g,px \
; RUN:     -r=%t2.bc,analias,px \
; RUN:     -r=%t2.bc,aliasee,px
; RUN: opt -function-import -import-all-index -enable-import-metadata -summary-file %t1.bc.thinlto.bc %t1.bc -o %t1.out
; RUN: opt -function-import -import-all-index -summary-file %t2.bc.thinlto.bc %t2.bc -o %t2.out
; RUN: llvm-dis -o - %t1.out | FileCheck %s --check-prefix=IMPORT
; RUN: llvm-dis -o - %t2.out | FileCheck %s --check-prefix=EXPORT

; Save the generated index files.
; RUN: cp %t1.bc.thinlto.bc %t1.bc.thinlto.bc.orig
; RUN: cp %t2.bc.thinlto.bc %t2.bc.thinlto.bc.orig

; Copy the minimized bitcode to the regular bitcode path so the module
; paths in the index are the same (save the regular bitcode for use again
; further down).
; RUN: cp %t1.bc %t1.bc.sv
; RUN: cp %t1.thinlink.bc %t1.bc
; RUN: cp %t2.bc %t2.bc.sv
; RUN: cp %t2.thinlink.bc %t2.bc

; Next perform the thin link on the minimized bitcode files, and compare dumps
; of the resulting indexes to the above dumps to ensure they are identical.
; RUN: rm -f %t1.bc.thinlto.bc %t2.bc.thinlto.bc
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.o -save-temps \
; RUN:     -thinlto-distributed-indexes \
; RUN:     -r=%t1.bc,g, \
; RUN:     -r=%t1.bc,analias, \
; RUN:     -r=%t1.bc,f,px \
; RUN:     -r=%t2.bc,g,px \
; RUN:     -r=%t2.bc,analias,px \
; RUN:     -r=%t2.bc,aliasee,px
; RUN: diff %t1.bc.thinlto.bc.orig %t1.bc.thinlto.bc
; RUN: diff %t2.bc.thinlto.bc.orig %t2.bc.thinlto.bc

; Make sure importing occurs as expected
; RUN: cp %t1.bc.sv %t1.bc
; RUN: cp %t2.bc.sv %t2.bc
; RUN: opt -function-import -import-all-index -enable-import-metadata -summary-file %t1.bc.thinlto.bc %t1.bc -o %t1.out
; RUN: opt -function-import -import-all-index -summary-file %t2.bc.thinlto.bc %t2.bc -o %t2.out
; RUN: llvm-dis -o - %t1.out | FileCheck %s --check-prefix=IMPORT
; RUN: llvm-dis -o - %t2.out | FileCheck %s --check-prefix=EXPORT

; IMPORT: define available_externally i32 @g() !thinlto_src_module
; IMPORT: define available_externally void @analias() !thinlto_src_module
; EXPORT: @G.llvm.

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare i32 @g(...)
declare void @analias(...)

define void @f() {
entry:
  call i32 (...) @g()
  call void (...) @analias()
  ret void
}

; THINLINKBITCODE-NOT: IDENTIFICATION_BLOCK_ID
; THINLINKBITCODE-NOT: BLOCKINFO_BLOCK
; THINLINKBITCODE-NOT: TYPE_BLOCK_ID
; THINLINKBITCODE-NOT: VSTOFFSET
; THINLINKBITCODE-NOT: CONSTANTS_BLOCK
; THINLINKBITCODE-NOT: METADATA_KIND_BLOCK
; THINLINKBITCODE-NOT: METADATA_BLOCK
; THINLINKBITCODE-NOT: OPERAND_BUNDLE_TAGS_BLOCK
; THINLINKBITCODE-NOT: UnknownBlock26
; THINLINKBITCODE-NOT: FUNCTION_BLOCK
; THINLINKBITCODE-NOT: VALUE_SYMTAB
; THINLINKBITCODE: MODULE_BLOCK
; THINLINKBITCODE: VERSION
; THINLINKBITCODE: SOURCE_FILENAME
; THINLINKBITCODE: GLOBALVAL_SUMMARY_BLOCK
; THINLINKBITCODE: HASH
; THINLINKBITCODE: SYMTAB_BLOCK
; THINLINKBITCODE: STRTAB_BLOCK

!llvm.dbg.cu = !{}

!1 = !{i32 2, !"Debug Info Version", i32 3}
!llvm.module.flags = !{!1}
