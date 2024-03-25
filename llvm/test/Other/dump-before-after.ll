; RUN: rm -rf %t/logs

; Basic dump before and after a single module pass

; RUN: opt %s -disable-output -passes='no-op-module' -ir-dump-directory %t/logs -print-after=no-op-module -print-before=no-op-module
; RUN: ls %t/logs | FileCheck %s --check-prefix=SINGLE-PASS
; RUN: ls %t/logs | count 2
; SINGLE-PASS-DAG: 0-[[MODULE_NAME_HASH:[a-z0-9]+]]-module-NoOpModulePass-after.ll
; SINGLE-PASS-DAG: 0-[[MODULE_NAME_HASH]]-module-NoOpModulePass-before.ll
; RUN: cat %t/logs/*after.ll | FileCheck %s --check-prefix=SINGLE-PASS-CONTENTS

; SINGLE-PASS-CONTENTS: ; *** IR Dump After NoOpModulePass on [module] ***
; SINGLE-PASS-CONTENTS: define void @foo() {
; SINGLE-PASS-CONTENTS:   ret void
; SINGLE-PASS-CONTENTS: }
; SINGLE-PASS-CONTENTS: define void @bar() {
; SINGLE-PASS-CONTENTS: entry:
; SINGLE-PASS-CONTENTS:   br label %my-loop
; SINGLE-PASS-CONTENTS: my-loop:                                          ; preds = %my-loop, %entry
; SINGLE-PASS-CONTENTS:   br label %my-loop
; SINGLE-PASS-CONTENTS: }

; RUN: rm -rf %t/logs

; Dump before and after multiple runs of the same module pass
; The integers preceeding log files represent relative pass execution order,
; but they are not necessarily continuous. That is passes which are run
; but not printed, still increment the count -- leading to gaps in the printed
; integers.

; RUN: opt %s -disable-output -passes='no-op-module,no-op-module,no-op-module' -ir-dump-directory %t/logs -print-after=no-op-module -print-before=no-op-module
; RUN: ls %t/logs | FileCheck %s --check-prefix=MULTIPLE-PASSES
; RUN: ls %t/logs | count 6
; MULTIPLE-PASSES-DAG: 0-[[MODULE_NAME_HASH:[a-z0-9]+]]-module-NoOpModulePass-after.ll
; MULTIPLE-PASSES-DAG: 0-[[MODULE_NAME_HASH]]-module-NoOpModulePass-before.ll
; MULTIPLE-PASSES-DAG: 1-[[MODULE_NAME_HASH]]-module-NoOpModulePass-after.ll
; MULTIPLE-PASSES-DAG: 1-[[MODULE_NAME_HASH]]-module-NoOpModulePass-before.ll
; MULTIPLE-PASSES-DAG: 2-[[MODULE_NAME_HASH]]-module-NoOpModulePass-after.ll
; MULTIPLE-PASSES-DAG: 2-[[MODULE_NAME_HASH]]-module-NoOpModulePass-before.ll
; RUN: rm -rf %t/logs

; Dump before and after multiple passes, of various levels of granularity

; RUN: opt %s -disable-output -passes='no-op-module,cgscc(no-op-cgscc),function(no-op-function),function(loop(no-op-loop)),no-op-module' -ir-dump-directory %t/logs -print-after=no-op-module,no-op-cgscc,no-op-function,no-op-loop -print-before=no-op-module,no-op-cgscc,no-op-function,no-op-loop
; RUN: ls %t/logs | FileCheck %s --check-prefix=MULTIPLE-GRANULAR-PASSES
; RUN: ls %t/logs | count 14
; MULTIPLE-GRANULAR-PASSES-DAG: 0-[[MODULE_NAME_HASH:[a-z0-9]+]]-module-NoOpModulePass-after.ll
; MULTIPLE-GRANULAR-PASSES-DAG: 0-[[MODULE_NAME_HASH]]-module-NoOpModulePass-before.ll
; MULTIPLE-GRANULAR-PASSES-DAG: 1-[[MODULE_NAME_HASH]]-scc-[[SCC_FOO_HASH:[a-z0-9]+]]-NoOpCGSCCPass-after.ll
; MULTIPLE-GRANULAR-PASSES-DAG: 1-[[MODULE_NAME_HASH]]-scc-[[SCC_FOO_HASH]]-NoOpCGSCCPass-before.ll
; MULTIPLE-GRANULAR-PASSES-DAG: 2-[[MODULE_NAME_HASH]]-scc-[[SCC_BAR_HASH:[a-z0-9]+]]-NoOpCGSCCPass-after.ll
; MULTIPLE-GRANULAR-PASSES-DAG: 2-[[MODULE_NAME_HASH]]-scc-[[SCC_BAR_HASH]]-NoOpCGSCCPass-before.ll
; MULTIPLE-GRANULAR-PASSES-DAG: 3-[[MODULE_NAME_HASH]]-function-[[FUNCTION_FOO_HASH:[a-z0-9]+]]-NoOpFunctionPass-after.ll
; MULTIPLE-GRANULAR-PASSES-DAG: 3-[[MODULE_NAME_HASH]]-function-[[FUNCTION_FOO_HASH]]-NoOpFunctionPass-before.ll
; MULTIPLE-GRANULAR-PASSES-DAG: 4-[[MODULE_NAME_HASH]]-function-[[FUNCTION_BAR_HASH:[a-z0-9]+]]-NoOpFunctionPass-after.ll
; MULTIPLE-GRANULAR-PASSES-DAG: 4-[[MODULE_NAME_HASH]]-function-[[FUNCTION_BAR_HASH]]-NoOpFunctionPass-before.ll
; MULTIPLE-GRANULAR-PASSES-DAG: 9-[[MODULE_NAME_HASH]]-loop-[[LOOP_NAME_HASH:[a-z0-9]+]]-NoOpLoopPass-after.ll
; MULTIPLE-GRANULAR-PASSES-DAG: 9-[[MODULE_NAME_HASH]]-loop-[[LOOP_NAME_HASH]]-NoOpLoopPass-before.ll
; MULTIPLE-GRANULAR-PASSES-DAG: 10-[[MODULE_NAME_HASH]]-module-NoOpModulePass-after.ll
; MULTIPLE-GRANULAR-PASSES-DAG: 10-[[MODULE_NAME_HASH]]-module-NoOpModulePass-before.ll
; RUN: rm -rf %t/logs

define void @foo() {
    ret void
}

define void @bar() {
entry:
    br label %my-loop
my-loop:
    br label %my-loop
}
