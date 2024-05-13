; Test plugin options for opt-remarks.
; RUN: llvm-as %s -o %t.o
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext -shared \
; RUN:	  -plugin-opt=save-temps \
; RUN:    -plugin-opt=opt-remarks-passes=inline \
; RUN:    -plugin-opt=opt-remarks-format=yaml \
; RUN:    -plugin-opt=opt-remarks-filename=%t.yaml %t.o -o %t2.o 2>&1
; RUN: llvm-dis %t2.o.0.4.opt.bc -o - | FileCheck %s
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext -shared \
; RUN:    -plugin-opt=opt-remarks-passes=inline \
; RUN:    -plugin-opt=opt-remarks-format=yaml \
; RUN:    -plugin-opt=opt-remarks-with-hotness \
; RUN:	  -plugin-opt=opt-remarks-filename=%t.hot.yaml %t.o -o %t2.o 2>&1
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext -shared \
; RUN:    -plugin-opt=opt-remarks-passes=inline \
; RUN:    -plugin-opt=opt-remarks-format=yaml \
; RUN:    -plugin-opt=opt-remarks-with-hotness \
; RUN:    -plugin-opt=opt-remarks-hotness-threshold=300 \
; RUN:	  -plugin-opt=opt-remarks-filename=%t.t300.yaml %t.o -o %t2.o 2>&1
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext -shared \
; RUN:    -plugin-opt=opt-remarks-passes=inline \
; RUN:    -plugin-opt=opt-remarks-format=yaml \
; RUN:    -plugin-opt=opt-remarks-with-hotness \
; RUN:    -plugin-opt=opt-remarks-hotness-threshold=301 \
; RUN:	  -plugin-opt=opt-remarks-filename=%t.t301.yaml %t.o -o %t2.o 2>&1
; RUN: cat %t.yaml | FileCheck %s -check-prefix=YAML
; RUN: cat %t.hot.yaml | FileCheck %s -check-prefix=YAML-HOT
; RUN: FileCheck %s -check-prefix=YAML-HOT < %t.t300.yaml
; RUN: count 0 < %t.t301.yaml

; Check that @f is inlined after optimizations.
; CHECK-LABEL: define i32 @_start
; CHECK-NEXT:  %a.i = tail call i32 @bar()
; CHECK-NEXT:  ret i32 %a.i
; CHECK-NEXT: }

; YAML: --- !Missed
; YAML-NEXT: Pass:            inline
; YAML-NEXT: Name:            NoDefinition
; YAML-NEXT: Function:        f
; YAML-NEXT: Args:
; YAML-NEXT:   - Callee:          bar
; YAML-NEXT:   - String:          ' will not be inlined into '
; YAML-NEXT:   - Caller:          f
; YAML-NEXT:   - String:          ' because its definition is unavailable'
; YAML-NEXT: ...
; YAML-NEXT: --- !Missed
; YAML-NEXT: Pass:            inline
; YAML-NEXT: Name:            NoDefinition
; YAML-NEXT: Function:        f
; YAML-NEXT: Args:
; YAML-NEXT:   - Callee:          bar
; YAML-NEXT:   - String:          ' will not be inlined into '
; YAML-NEXT:   - Caller:          f
; YAML-NEXT:   - String:          ' because its definition is unavailable'
; YAML-NEXT: ...
; YAML-NEXT: --- !Passed
; YAML-NEXT: Pass:            inline
; YAML-NEXT: Name:            Inlined
; YAML-NEXT: Function:        _start
; YAML-NEXT: Args:
; YAML-NEXT:   - String:          ''''
; YAML-NEXT:   - Callee:          f
; YAML-NEXT:   - String:          ''' inlined into '''
; YAML-NEXT:   - Caller:          _start
; YAML-NEXT:   - String:          ''''
; YAML-NEXT:   - String:          ' with '
; YAML-NEXT:   - String:          '(cost='
; YAML-NEXT:   - Cost:            '0'
; YAML-NEXT:   - String:          ', threshold='
; YAML-NEXT:   - Threshold:       '337'
; YAML-NEXT:   - String:          ')'
; YAML-NEXT: ...

; YAML-HOT: --- !Passed
; YAML-HOT: Pass:            inline
; YAML-HOT-NEXT: Name:            Inlined
; YAML-HOT-NEXT: Function:        _start
; YAML-HOT-NEXT: Hotness:         300
; YAML-HOT-NEXT: Args:
; YAML-HOT-NEXT:   - String:          ''''
; YAML-HOT-NEXT:   - Callee:          f
; YAML-HOT-NEXT:   - String:          ''' inlined into '''
; YAML-HOT-NEXT:   - Caller:          _start
; YAML-HOT-NEXT:   - String:          ''''
; YAML-HOT-NEXT:   - String:          ' with '
; YAML-HOT-NEXT:   - String:          '(cost='
; YAML-HOT-NEXT:   - Cost:            '0'
; YAML-HOT-NEXT:   - String:          ', threshold='
; YAML-HOT-NEXT:   - Threshold:       '337'
; YAML-HOT-NEXT:   - String:          ')'
; YAML-HOT-NEXT: ...

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @bar()

define i32 @f() {
  %a = call i32 @bar()
  ret i32 %a
}

define i32 @_start() !prof !0 {
  %call = call i32 @f()
  ret i32 %call
}

!0 = !{!"function_entry_count", i64 300}
