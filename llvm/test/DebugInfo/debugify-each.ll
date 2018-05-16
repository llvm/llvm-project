; RUN: opt -debugify-each -O3 -S -o /dev/null < %s > %t
; RUN: FileCheck %s -input-file=%t -check-prefix=MODULE-PASS
; RUN: FileCheck %s -input-file=%t -check-prefix=FUNCTION-PASS

; RUN: opt -enable-debugify -debugify-each -O3 -S -o /dev/null < %s > %t
; RUN: FileCheck %s -input-file=%t -check-prefix=MODULE-PASS
; RUN: FileCheck %s -input-file=%t -check-prefix=FUNCTION-PASS

; RUN: opt -debugify-each -instrprof -instrprof -sroa -sccp -S -o /dev/null < %s > %t
; RUN: FileCheck %s -input-file=%t -check-prefix=MODULE-PASS
; RUN: FileCheck %s -input-file=%t -check-prefix=FUNCTION-PASS

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}

; Verify that the module & function (check-)debugify passes run at least twice.

; MODULE-PASS: CheckModuleDebugify [{{.*}}]
; MODULE-PASS: CheckModuleDebugify [{{.*}}]

; FUNCTION-PASS: CheckFunctionDebugify [{{.*}}]
; FUNCTION-PASS: CheckFunctionDebugify [{{.*}}]
; FUNCTION-PASS: CheckFunctionDebugify [{{.*}}]
; FUNCTION-PASS: CheckFunctionDebugify [{{.*}}]
