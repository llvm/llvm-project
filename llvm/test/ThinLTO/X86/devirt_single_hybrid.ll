; Check that we import and inline virtual method with single implementation
; when we're running hybrid LTO.
;
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %s -o %t-main.bc
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %p/Inputs/devirt_single_hybrid_foo.ll -o %t-foo.bc
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %p/Inputs/devirt_single_hybrid_bar.ll -o %t-bar.bc
; RUN: llvm-lto2 run -save-temps %t-main.bc %t-foo.bc %t-bar.bc -pass-remarks=. -o %t \
; RUN:    -r=%t-foo.bc,_Z3fooP1A,pl \
; RUN:    -r=%t-main.bc,main,plx \
; RUN:    -r=%t-main.bc,_Z3barv,l \
; RUN:    -r=%t-bar.bc,_Z3barv,pl \
; RUN:    -r=%t-bar.bc,_Z3fooP1A, \
; RUN:    -r=%t-bar.bc,_ZNK1A1fEv,pl \
; RUN:    -r=%t-bar.bc,_ZTV1A,l \
; RUN:    -r=%t-bar.bc,_ZTVN10__cxxabiv117__class_type_infoE, \
; RUN:    -r=%t-bar.bc,_ZTS1A,pl \
; RUN:    -r=%t-bar.bc,_ZTI1A,pl \
; RUN:    -r=%t-bar.bc,_ZNK1A1fEv, \
; RUN:    -r=%t-bar.bc,_ZTV1A,pl \
; RUN:    -r=%t-bar.bc,_ZTI1A, 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t.1.3.import.bc -o - | FileCheck %s --check-prefix=IMPORT
; RUN: llvm-dis %t.1.5.precodegen.bc -o - | FileCheck %s --check-prefix=CODEGEN

; REMARK-COUNT-3: single-impl: devirtualized a call to _ZNK1A1fEv

; IMPORT:       define available_externally hidden i32 @_ZNK1A1fEv(%struct.A* %this)
; IMPORT-NEXT:  entry:
; IMPORT-NEXT:      ret i32 3

; CODEGEN:        define hidden i32 @main()
; CODEGEN-NEXT:   entry:
; CODEGEN-NEXT:     ret i32 23

; Virtual method should have been optimized out
; CODEGEN-NOT: _ZNK1A1fEv

; ModuleID = 'main.cpp'
source_filename = "main.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse uwtable
define hidden i32 @main() local_unnamed_addr {
entry:
  %call = tail call i32 @_Z3barv()
  ret i32 %call
}

declare dso_local i32 @_Z3barv() local_unnamed_addr

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 (trunk 373596)"}
