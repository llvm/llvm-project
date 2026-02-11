; Make sure that -print-before-pass-number works to write to a
; temporary file, and not stderr, when combined with
; -ir-dump-directory

; RUN_PASS_NUMBERS: Running pass 1 NoOpModulePass on [module]
; RUN_PASS_NUMBERS: Running pass 2 NoOpModulePass on [module]
; RUN_PASS_NUMBERS: Running pass 3 NoOpModulePass on [module]


; RUN: rm -rf %t/logs
; RUN: opt %s -disable-output -passes='no-op-module,no-op-module,no-op-module' -print-before-pass-number=2 -print-pass-numbers -ir-dump-directory %t/logs 2>&1 | FileCheck -check-prefix=RUN_PASS_NUMBERS %s
; RUN: ls %t/logs | FileCheck --check-prefix=BEFORE2 %s
; RUN: ls %t/logs | count 1
; BEFORE2: 2-[[MODULE_NAME_HASH:[a-z0-9]+]]-module-NoOpModulePass-before.ll

; RUN: cat %t/logs/* | FileCheck -check-prefix=BEFORE2_COMMENT %s
; BEFORE2_COMMENT: ; *** IR Dump Before 2-NoOpModulePass on [module] ***



; RUN: rm -rf %t/logs
; RUN: opt %s -disable-output -passes='no-op-module,no-op-module,no-op-module' -print-before-pass-number=1 -print-pass-numbers -ir-dump-directory %t/logs 2>&1 | FileCheck -check-prefix=RUN_PASS_NUMBERS %s
; RUN: ls %t/logs | FileCheck --check-prefix=BEFORE1 %s
; RUN: ls %t/logs | count 1

; BEFORE1: 1-[[MODULE_NAME_HASH:[a-z0-9]+]]-module-NoOpModulePass-before.ll

; RUN: cat %t/logs/* | FileCheck -check-prefix=BEFORE1_COMMENT %s
; BEFORE1_COMMENT: ; *** IR Dump Before 1-NoOpModulePass on [module] ***



; RUN: rm -rf %t/logs
; RUN: opt %s -disable-output -passes='no-op-module,no-op-module,no-op-module' -print-after-pass-number=2 -print-pass-numbers -ir-dump-directory %t/logs 2>&1 | FileCheck -check-prefix=RUN_PASS_NUMBERS %s
; RUN: ls %t/logs | FileCheck --check-prefix=AFTER2 %s
; RUN: ls %t/logs | count 1
; AFTER2: 2-[[MODULE_NAME_HASH:[a-z0-9]+]]-module-NoOpModulePass-after.ll


; RUN: cat %t/logs/* | FileCheck -check-prefix=AFTER2_COMMENT %s
; AFTER2_COMMENT: ; *** IR Dump After 2-NoOpModulePass on [module] ***



; RUN: rm -rf %t/logs
; RUN: opt %s -disable-output -passes='no-op-module,no-op-module,no-op-module' -print-after-pass-number=1 -print-pass-numbers -ir-dump-directory %t/logs 2>&1 | FileCheck -check-prefix=RUN_PASS_NUMBERS %s
; RUN: ls %t/logs | FileCheck --check-prefix=AFTER1 %s
; RUN: ls %t/logs | count 1
; AFTER1: 1-[[MODULE_NAME_HASH:[a-z0-9]+]]-module-NoOpModulePass-after.ll


; RUN: cat %t/logs/* | FileCheck -check-prefix=AFTER1_COMMENT %s
; AFTER1_COMMENT: ; *** IR Dump After 1-NoOpModulePass on [module] ***

define void @foo() {
  ret void
}

define void @bar() {
entry:
  br label %my-loop

my-loop:
  br label %my-loop
}
