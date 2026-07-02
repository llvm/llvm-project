; RUN: opt -S -print-changed=quiet -passes=function-attrs 2>&1 \
; RUN:   -o /dev/null < %s | FileCheck %s --check-prefix=FULL
; RUN: opt -S -print-changed=quiet,hash -passes=function-attrs 2>&1 \
; RUN:   -o /dev/null < %s | FileCheck %s --check-prefix=FULL
; RUN: opt -S -print-changed=attrs-only -passes=function-attrs 2>&1 \
; RUN:   -o /dev/null < %s | FileCheck %s --check-prefix=ATTRS
; RUN: opt -S -print-changed=quiet,attrs-only -passes=function-attrs 2>&1 \
; RUN:   -o /dev/null < %s | FileCheck %s --check-prefix=ATTRS
; RUN: opt -S -print-changed=attrs-only -passes=instsimplify 2>&1 \
; RUN:   -o /dev/null < %s | FileCheck %s --check-prefix=TEXT

define i32 @f() {
entry:
  %sum = add i32 2, 3
  ret i32 %sum
}

; FULL: *** IR Dump After {{.*}}FunctionAttrsPass on (f) ***
; FULL: Function Attrs: {{.*}}memory(none)
; FULL: define {{.*}}i32 @f()
; FULL: ret i32 %sum

; ATTRS: *** IR Attribute Changes After {{.*}}FunctionAttrsPass on (f) ***
; ATTRS-NEXT: Function: f
; ATTRS-NEXT:   function before: (none)
; ATTRS-NEXT:   function after:  {{.*}}memory(none)
; ATTRS-NEXT:   return before: (none)
; ATTRS-NEXT:   return after:  noundef
; ATTRS-NOT: ret i32

; TEXT: *** IR Dump After InstSimplifyPass on f ***
; TEXT: ret i32 5
