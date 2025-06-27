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

; MISSING-OPT-LEVEL: invalid optimization level ''
; INVALID-OPT-LEVEL: invalid optimization level 'foo'
