; RUN: opt -disable-output -print-pipeline-passes \
; RUN:   -passes-ep-vectorizer-start='no-op-function' \
; RUN:   -passes='function(vectorizer-start-callbacks<O3>)' < %s 2>&1 | FileCheck %s --check-prefix=VECSTART
; RUN: opt -disable-output -print-pipeline-passes \
; RUN:   -passes-ep-peephole='no-op-function' \
; RUN:   -passes='peephole-callbacks<Os>' < %s 2>&1 | FileCheck %s --check-prefix=PEEP
; RUN: opt -disable-output -print-pipeline-passes \
; RUN:   -passes-ep-pipeline-start='no-op-module' \
; RUN:   -passes='pipeline-start-callbacks<O1>' < %s 2>&1 | FileCheck %s --check-prefix=MODSTART
; RUN: opt -disable-output -print-pipeline-passes \
; RUN:   -passes-ep-pipeline-early-simplification='no-op-module' \
; RUN:   -passes='pipeline-early-simplification-callbacks<O2>' < %s 2>&1 | FileCheck %s --check-prefix=LTOEARLY
; RUN: opt -disable-output -print-pipeline-passes \
; RUN:   -passes-ep-optimizer-early='no-op-module' \
; RUN:   -passes='optimizer-early-callbacks<O2>' < %s 2>&1 | FileCheck %s --check-prefix=OPTEARLY
; RUN: opt -disable-output -print-pipeline-passes \
; RUN:   -passes-ep-optimizer-last='no-op-module' \
; RUN:   -passes='optimizer-last-callbacks<O2>' < %s 2>&1 | FileCheck %s --check-prefix=OPTLAST
; RUN: opt -disable-output -print-pipeline-passes \
; RUN:   -passes-ep-scalar-optimizer-late='no-op-function' \
; RUN:   -passes='function(scalar-optimizer-late-callbacks<O2>)' < %s 2>&1 | FileCheck %s --check-prefix=SCALATE
; RUN: opt -disable-output -print-pipeline-passes \
; RUN:   -passes-ep-vectorizer-end='no-op-function' \
; RUN:   -passes='function(vectorizer-end-callbacks<O3>)' < %s 2>&1 | FileCheck %s --check-prefix=VECEND
; RUN: opt -disable-output -print-pipeline-passes \
; RUN:   -passes-ep-late-loop-optimizations='no-op-loop' \
; RUN:   -passes='loop(late-loop-optimizations-callbacks<O2>)' < %s 2>&1 | FileCheck %s --check-prefix=LATELOOP
; RUN: opt -disable-output -print-pipeline-passes \
; RUN:   -passes-ep-loop-optimizer-end='no-op-loop' \
; RUN:   -passes='loop(loop-optimizer-end-callbacks<O2>)' < %s 2>&1 | FileCheck %s --check-prefix=LOOPOPTEND
; RUN: opt -disable-output -print-pipeline-passes \
; RUN:   -passes-ep-cgscc-optimizer-late='no-op-cgscc' \
; RUN:   -passes='cgscc(cgscc-optimizer-late-callbacks<O2>)' < %s 2>&1 | FileCheck %s --check-prefix=CGSCCTLATE
; RUN: not opt -disable-output -passes='vectorizer-start-callbacks<foo>' < %s 2>&1 | FileCheck %s --check-prefix=INVALID

; VECSTART: no-op-function
; PEEP: no-op-function
; MODSTART: no-op-module
; LTOEARLY: no-op-module
; OPTEARLY: no-op-module
; OPTLAST: no-op-module
; SCALATE: no-op-function
; VECEND: no-op-function
; LATELOOP: no-op-loop
; LOOPOPTEND: no-op-loop
; CGSCCTLATE: no-op-cgscc
; INVALID: invalid optimization level 'foo'

define void @f() {
entry:
  ret void
}
