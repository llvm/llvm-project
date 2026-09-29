; RUN: split-file %s %t
; RUN: not llvm-as < %t/parse-error-1.ll -o /dev/null 2>&1 | FileCheck --check-prefix=PARSE-ERROR-1 %s
; RUN: not llvm-as < %t/parse-error-2.ll -o /dev/null 2>&1 | FileCheck --check-prefix=PARSE-ERROR-2 %s
; RUN: not llvm-as < %t/parse-error-3.ll -o /dev/null 2>&1 | FileCheck --check-prefix=PARSE-ERROR-3 %s
; RUN: not llvm-as < %t/parse-error-4.ll -o /dev/null 2>&1 | FileCheck --check-prefix=PARSE-ERROR-4 %s
; RUN: not llvm-as < %t/end-not-larger-start.ll -o /dev/null 2>&1 | FileCheck --check-prefix=END-NOT-LARGER-START %s

;--- parse-error-1.ll

; PARSE-ERROR-1: error: expected integer
@g = external global i8
define ptr @test() {
  ret ptr getelementptr inrange (i8, ptr @g, i64 8)
}

;--- parse-error-2.ll

; PARSE-ERROR-2: error: expected ','
@g = external global i8
define ptr @test() {
  ret ptr getelementptr inrange(42 (i8, ptr @g, i64 8)
}

;--- parse-error-3.ll

; PARSE-ERROR-3: error: expected integer
@g = external global i8
define ptr @test() {
  ret ptr getelementptr inrange(42, (i8, ptr @g, i64 8)
}

;--- parse-error-4.ll

; PARSE-ERROR-4: error: expected ')'
@g = external global i8
define ptr @test() {
  ret ptr getelementptr inrange(42, 123 (i8, ptr @g, i64 8)
}

;--- end-not-larger-start.ll

; END-NOT-LARGER-START: error: expected end to be larger than start
@g = external global i8
define ptr @test() {
  ret ptr getelementptr inrange(42, 42) (i8, ptr @g, i64 8)
}
