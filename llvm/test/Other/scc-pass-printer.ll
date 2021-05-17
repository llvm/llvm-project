; RUN: opt -enable-new-pm=0 < %s 2>&1 -disable-output \
; RUN: 	   -inline -print-after-all | FileCheck %s --check-prefix=LEGACY
; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes=inline -print-after-all | FileCheck %s -check-prefix=INL
; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes=inliner-wrapper -print-after-all | FileCheck %s -check-prefix=INL
; RUN: opt -enable-new-pm=0 < %s 2>&1 -disable-output \
; RUN: 	   -inline -print-after-all -print-module-scope | FileCheck %s -check-prefix=LEGACY-MOD
; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes=inline -print-after-all -print-module-scope | FileCheck %s -check-prefix=INL-MOD
; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes=inliner-wrapper -print-after-all -print-module-scope | FileCheck %s -check-prefix=INL-MOD

; LEGACY: IR Dump After Function Integration/Inlining
; LEGACY:        define void @bar()
; LEGACY-NEXT:   call void @foo()
; LEGACY: define void @foo()
; LEGACY-NEXT:   call void @bar()
; LEGACY: IR Dump After Function Integration/Inlining
; LEGACY: define void @tester()
; LEGACY-NEXT:  call void @foo()

; INL:      IR Dump After InlinerPass on (foo, bar) ***
; INL:      define void @foo()
; INL-NEXT:   call void @bar()
; INL:      define void @bar()
; INL-NEXT:   call void @foo()
; INL:      IR Dump After InlinerPass on (tester) ***
; INL:      define void @tester()
; INL-NEXT:   call void @foo()

; LEGACY-MOD:      IR Dump After Function Integration/Inlining
; LEGACY-MOD-NEXT: ModuleID =
; LEGACY-MOD:      define void @tester()
; LEGACY-MOD:      define void @foo()
; LEGACY-MOD:      define void @bar()

; INL-MOD-LABEL:*** IR Dump After InlinerPass on (foo, bar) ***
; INL-MOD-NEXT: ModuleID =
; INL-MOD-NEXT: source_filename =
; INL-MOD: define void @tester()
; INL-MOD-NEXT:  call void @foo()
; INL-MOD: define void @foo()
; INL-MOD-NEXT:  call void @bar()
; INL-MOD: define void @bar()
; INL-MOD-NEXT:   call void @foo()
; INL-MOD-LABEL:*** IR Dump After InlinerPass on (tester) ***
; INL-MOD-NEXT: ModuleID =
; INL-MOD-NEXT: source_filename =
; INL-MOD: define void @tester()
; INL-MOD-NEXT:  call void @foo()
; INL-MOD: define void @foo()
; INL-MOD-NEXT:   call void @bar()
; INL-MOD: define void @bar()
; INL-MOD-NEXT:  call void @foo()
; INL-MOD: IR Dump After
; INL-MOD-NEXT: ModuleID =
; INL-MOD-NEXT: source_filename =
; INL-MOD-NOT: Printing <null> Function

define void @tester() noinline {
  call void @foo()
  ret void
}

define void @foo() noinline {
  call void @bar()
  ret void
}

define void @bar() noinline {
  call void @foo()
  ret void
}
