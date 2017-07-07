; Check per module hash.
; RUN: opt  -module-hash  %s -o - | llvm-bcanalyzer -dump -check-hash=foo | FileCheck %s --check-prefix=MOD1
; MOD1: <HASH op0={{[0-9]*}} op1={{[0-9]*}} op2={{[0-9]*}} op3={{[0-9]*}} op4={{[0-9]*}} (match)/>
; RUN: opt  -module-hash  %p/Inputs/module_hash.ll -o - | llvm-bcanalyzer -dump -check-hash=bar | FileCheck %s --check-prefix=MOD2
; MOD2: <HASH op0={{[0-9]*}} op1={{[0-9]*}} op2={{[0-9]*}} op3={{[0-9]*}} op4={{[0-9]*}} (match)/>

; Check that the hash matches in the combined index.

; First regenerate the modules with a summary
; RUN: opt  -module-hash -module-summary %s -o %t.m1.bc
; RUN: opt  -module-hash -module-summary %p/Inputs/module_hash.ll -o %t.m2.bc

; Recover the hashes from the modules themselves.
; RUN: llvm-bcanalyzer -dump %t.m1.bc | grep '<HASH'  > %t.hash
; RUN: llvm-bcanalyzer -dump %t.m2.bc | grep '<HASH'  >> %t.hash

; Generate the combined index and gather the hashes there.
; RUN: llvm-lto --thinlto-action=thinlink -o - %t.m1.bc %t.m2.bc | llvm-bcanalyzer -dump  | grep '<HASH ' >> %t.hash

; Validate the output now, the hahes in the individual modules and the combined index are in the same file.
; RUN: cat %t.hash | FileCheck %s --check-prefix=COMBINED

; First capture the value of the hash for the two modules.
; COMBINED: <HASH op0=[[HASH1_1:[0-9]*]] op1=[[HASH1_2:[0-9]*]] op2=[[HASH1_3:[0-9]*]] op3=[[HASH1_4:[0-9]*]] op4=[[HASH1_5:[0-9]*]]/>
; COMBINED: <HASH op0=[[HASH2_1:[0-9]*]] op1=[[HASH2_2:[0-9]*]] op2=[[HASH2_3:[0-9]*]] op3=[[HASH2_4:[0-9]*]] op4=[[HASH2_5:[0-9]*]]/>

; Validate against the value extracted from the combined index
; COMBINED-DAG: <HASH abbrevid={{[0-9]*}} op0=[[HASH1_1]] op1=[[HASH1_2]] op2=[[HASH1_3]] op3=[[HASH1_4]] op4=[[HASH1_5]]/>
; COMBINED-DAG: <HASH abbrevid={{[0-9]*}} op0=[[HASH2_1]] op1=[[HASH2_2]] op2=[[HASH2_3]] op3=[[HASH2_4]] op4=[[HASH2_5]]/>


; Need a function for the combined index to be populated.
define void @foo() {
    ret void
}
