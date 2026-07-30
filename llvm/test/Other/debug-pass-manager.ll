; RUN: opt -passes=inline < %s -debug-pass-manager 2>&1 | FileCheck %s --check-prefix=NORMAL
; RUN: opt -passes=inline < %s -debug-pass-manager=quiet 2>&1 | FileCheck %s --check-prefix=QUIET
; RUN: opt -passes=inline < %s -debug-pass-manager=verbose 2>&1 | FileCheck %s --check-prefix=VERBOSE

define void @a() {
  ret void
}

; QUIET-NOT: Running pass: ModuleToPostOrderCGSCCPassAdaptor
; QUIET: Running pass: InlinerPass
; QUIET-NOT: Running analysis

; NORMAL-NOT: Running pass: ModuleToPostOrderCGSCCPassAdaptor
; NORMAL: Running pass: InlinerPass
; NORMAL: Running analysis

; VERBOSE: Running pass: ModuleToPostOrderCGSCCPassAdaptor
; VERBOSE: Running pass: InlinerPass
; VERBOSE: Running analysis
