; First ensure that the ThinLTO handling in the gold plugin handles
; bitcode without summary sections gracefully.
; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/thinlto.ll -o %t2.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=thinlto-index-only \
; RUN:    -shared %t.o %t2.o -o %t3
; RUN: not test -e %t3
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=thinlto \
; RUN:    -shared %t.o %t2.o -o %t4
; RUN: llvm-nm %t4 | FileCheck %s --check-prefix=NM

; Next generate summary sections and test gold handling.
; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o %t2.o

; Ensure gold generates an index and not a binary if requested.
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=thinlto-index-only \
; RUN:    -shared %t.o %t2.o -o %t3
; RUN: llvm-bcanalyzer -dump %t.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND1
; RUN: llvm-bcanalyzer -dump %t2.o.thinlto.bc | FileCheck %s --check-prefix=BACKEND2
; RUN: not test -e %t3

; Ensure gold generates an index as well as a binary with save-temps in ThinLTO mode.
; First force single-threaded mode
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=save-temps \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=jobs=1 \
; RUN:    -shared %t.o %t2.o -o %t4
; RUN: llvm-bcanalyzer -dump %t4.index.bc | FileCheck %s --check-prefix=COMBINED
; RUN: llvm-nm %t4 | FileCheck %s --check-prefix=NM

; Next force multi-threaded mode
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=save-temps \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=jobs=2 \
; RUN:    -shared %t.o %t2.o -o %t4
; RUN: llvm-bcanalyzer -dump %t4.index.bc | FileCheck %s --check-prefix=COMBINED
; RUN: llvm-nm %t4 | FileCheck %s --check-prefix=NM

; Test --plugin-opt=obj-path to ensure unique object files generated.
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=jobs=2 \
; RUN:    --plugin-opt=obj-path=%t5.o \
; RUN:    -shared %t.o %t2.o -o %t4
; RUN: llvm-nm %t5.o1 | FileCheck %s --check-prefix=NM2
; RUN: llvm-nm %t5.o2 | FileCheck %s --check-prefix=NM2

; NM: T f
; NM2: T {{f|g}}

; The backend index for this module contains summaries from itself and
; Inputs/thinlto.ll, as it imports from the latter.
; BACKEND1: <MODULE_STRTAB_BLOCK
; BACKEND1-NEXT: <ENTRY {{.*}} record string = '{{.*}}/test/tools/gold/X86/Output/thinlto.ll.tmp{{.*}}.o'
; BACKEND1-NEXT: <ENTRY {{.*}} record string = '{{.*}}/test/tools/gold/X86/Output/thinlto.ll.tmp{{.*}}.o'
; BACKEND1-NEXT: </MODULE_STRTAB_BLOCK
; BACKEND1-NEXT: <GLOBALVAL_SUMMARY_BLOCK
; BACKEND1-NEXT: <VERSION
; BACKEND1-NEXT: <COMBINED
; BACKEND1-NEXT: <COMBINED
; BACKEND1-NEXT: </GLOBALVAL_SUMMARY_BLOCK
; BACKEND1-NEXT: <VALUE_SYMTAB
; Check that the format is: op0=valueid, op1=offset, op2=funcguid,
; where funcguid is the lower 64 bits of the function name MD5.
; BACKEND1-NEXT: <COMBINED_ENTRY abbrevid={{[0-9]+}} op0={{1|2}} op1={{-3706093650706652785|-5300342847281564238}}
; BACKEND1-NEXT: <COMBINED_ENTRY abbrevid={{[0-9]+}} op0={{1|2}} op1={{-3706093650706652785|-5300342847281564238}}
; BACKEND1-NEXT: </VALUE_SYMTAB

; The backend index for Input/thinlto.ll contains summaries from itself only,
; as it does not import anything.
; BACKEND2: <MODULE_STRTAB_BLOCK
; BACKEND2-NEXT: <ENTRY {{.*}} record string = '{{.*}}/test/tools/gold/X86/Output/thinlto.ll.tmp2.o'
; BACKEND2-NEXT: </MODULE_STRTAB_BLOCK
; BACKEND2-NEXT: <GLOBALVAL_SUMMARY_BLOCK
; BACKEND2-NEXT: <VERSION
; BACKEND2-NEXT: <COMBINED
; BACKEND2-NEXT: </GLOBALVAL_SUMMARY_BLOCK
; BACKEND2-NEXT: <VALUE_SYMTAB
; Check that the format is: op0=valueid, op1=offset, op2=funcguid,
; where funcguid is the lower 64 bits of the function name MD5.
; BACKEND2-NEXT: <COMBINED_ENTRY abbrevid={{[0-9]+}} op0=1 op1=-5300342847281564238
; BACKEND2-NEXT: </VALUE_SYMTAB

; COMBINED: <MODULE_STRTAB_BLOCK
; COMBINED-NEXT: <ENTRY {{.*}} record string = '{{.*}}/test/tools/gold/X86/Output/thinlto.ll.tmp{{.*}}.o'
; COMBINED-NEXT: <ENTRY {{.*}} record string = '{{.*}}/test/tools/gold/X86/Output/thinlto.ll.tmp{{.*}}.o'
; COMBINED-NEXT: </MODULE_STRTAB_BLOCK
; COMBINED-NEXT: <GLOBALVAL_SUMMARY_BLOCK
; COMBINED-NEXT: <VERSION
; COMBINED-NEXT: <COMBINED
; COMBINED-NEXT: <COMBINED
; COMBINED-NEXT: </GLOBALVAL_SUMMARY_BLOCK
; COMBINED-NEXT: <VALUE_SYMTAB
; Check that the format is: op0=valueid, op1=offset, op2=funcguid,
; where funcguid is the lower 64 bits of the function name MD5.
; COMBINED-NEXT: <COMBINED_ENTRY abbrevid={{[0-9]+}} op0={{1|2}} op1={{-3706093650706652785|-5300342847281564238}}
; COMBINED-NEXT: <COMBINED_ENTRY abbrevid={{[0-9]+}} op0={{1|2}} op1={{-3706093650706652785|-5300342847281564238}}
; COMBINED-NEXT: </VALUE_SYMTAB

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
