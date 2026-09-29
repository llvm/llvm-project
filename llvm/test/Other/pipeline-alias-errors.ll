; RUN: not opt -passes="default" < %s 2>&1 | FileCheck %s --check-prefix=MISSING-OPT-LEVEL
; RUN: not opt -passes="default<foo>" < %s 2>&1 | FileCheck %s --check-prefix=INVALID-OPT-LEVEL
; RUN: not opt -passes="thinlto-pre-link" < %s 2>&1 | FileCheck %s --check-prefix=MISSING-OPT-LEVEL
; RUN: not opt -passes="thinlto-pre-link<foo>" < %s 2>&1 | FileCheck %s --check-prefix=INVALID-OPT-LEVEL
; RUN: not opt -passes="thinlto" < %s 2>&1 | FileCheck %s --check-prefix=MISSING-OPT-LEVEL
; RUN: not opt -passes="thinlto<foo>" < %s 2>&1 | FileCheck %s --check-prefix=INVALID-OPT-LEVEL
; RUN: not opt -passes="lto-pre-link" < %s 2>&1 | FileCheck %s --check-prefix=MISSING-OPT-LEVEL
; RUN: not opt -passes="lto-pre-link<foo>" < %s 2>&1 | FileCheck %s --check-prefix=INVALID-OPT-LEVEL
; RUN: not opt -passes="lto" < %s 2>&1 | FileCheck %s --check-prefix=MISSING-OPT-LEVEL
; RUN: not opt -passes="lto<foo>" < %s 2>&1 | FileCheck %s --check-prefix=INVALID-OPT-LEVEL
; RUN: not opt -passes="fatlto-pre-link" < %s 2>&1 | FileCheck %s --check-prefix=FATLTO-MISSING-OPT-LEVEL
; RUN: not opt -passes="fatlto-pre-link<foo>" < %s 2>&1 | FileCheck %s --check-prefix=FATLTO-INVALID-PARAMS

; MISSING-OPT-LEVEL: invalid optimization level ''
; INVALID-OPT-LEVEL: invalid optimization level 'foo'

; FATLTO-MISSING-OPT-LEVEL: missing optimization level for fatlto-pre-link pipeline
; FATLTO-INVALID-PARAMS: invalid fatlto-pre-link pass parameter 'foo'
