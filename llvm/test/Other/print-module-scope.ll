; This test is checking basic properties of -print-module-scope options:
;   - dumps all the module IR at once
;   - all the function attributes are shown, including those of declarations
;   - works on top of -print-after and -filter-print-funcs
;
; RUN: opt -enable-new-pm=0 < %s 2>&1 -disable-output \
; RUN: 	   -simplifycfg -print-after=simplifycfg -print-module-scope \
; RUN:	   | FileCheck %s -check-prefix=CFG
; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes=simplify-cfg -print-after-all -print-module-scope \
; RUN:	   | FileCheck %s -check-prefix=CFG
; RUN: opt -enable-new-pm=0 < %s 2>&1 -disable-output \
; RUN: 	   -simplifycfg -print-after=simplifycfg -filter-print-funcs=foo -print-module-scope \
; RUN:	   | FileCheck %s -check-prefix=FOO
; RUN: opt < %s 2>&1 -disable-output \
; RUN: 	   -passes=simplify-cfg -print-after-all -filter-print-funcs=foo -print-module-scope \
; RUN:	   | FileCheck %s -check-prefix=FOO

; CFG:      IR Dump After {{Simplify the CFG|SimplifyCFGPass}}
; CFG-SAME:   function: foo
; CFG-NEXT: ModuleID =
; CFG: define void @foo
; CFG: define void @bar
; CFG: declare void @baz
; CFG: IR Dump After
; CFG-SAME:   function: bar
; CFG-NEXT: ModuleID =
; CFG: define void @foo
; CFG: define void @bar
; CFG: declare void @baz

; FOO:      IR Dump After {{Simplify the CFG|SimplifyCFGPass}}
; FOO-NOT:    function: bar
; FOO-SAME:   function: foo
; FOO-NEXT: ModuleID =
; FOO:   Function Attrs: nounwind ssp
; FOO: define void @foo
; FOO:   Function Attrs: nounwind
; FOO: define void @bar
; FOO:   Function Attrs: nounwind readnone ssp
; FOO: declare void @baz

define void @foo() nounwind ssp {
  call void @baz()
  ret void
}

define void @bar() #0 {
  ret void
}

declare void @baz() #1

attributes #0 = { nounwind "frame-pointer"="all" }

attributes #1 = { nounwind readnone ssp "use-soft-float"="false" }
; FOO: attributes #{{[0-9]}} = { nounwind "frame-pointer"="all" }

; FOO: attributes #{{[0-9]}} = { nounwind readnone ssp "use-soft-float"="false" }

; FOO-NOT: IR Dump After {{Simplify the CFG|SimplifyCFGPass}}
