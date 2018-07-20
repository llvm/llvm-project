; RUN: opt -debugify-each -O3 -S -o /dev/null < %s 2> %t
; RUN: FileCheck %s -input-file=%t -check-prefix=MODULE-PASS
; RUN: FileCheck %s -input-file=%t -check-prefix=FUNCTION-PASS

; RUN: opt -enable-debugify -debugify-each -O3 -S -o /dev/null < %s 2> %t
; RUN: FileCheck %s -input-file=%t -check-prefix=MODULE-PASS
; RUN: FileCheck %s -input-file=%t -check-prefix=FUNCTION-PASS

; RUN: opt -debugify-each -instrprof -instrprof -sroa -sccp -S -o /dev/null < %s 2> %t
; RUN: FileCheck %s -input-file=%t -check-prefix=MODULE-PASS
; RUN: FileCheck %s -input-file=%t -check-prefix=FUNCTION-PASS

; Verify that debugify each can be safely used with piping
; RUN: opt -debugify-each -O1 < %s | opt -O2 -o /dev/null

; Check that the quiet mode emits no messages.
; RUN: opt -disable-output -debugify-quiet -debugify-each -O1 < %s 2>&1 | count 0

; Check that stripped textual IR compares equal before and after applying
; debugify.
; RUN: opt -O1 < %s -S -o - | \
; RUN:   opt -strip -strip-dead-prototypes -strip-named-metadata -S -o %t.before
; RUN: opt -O1 -debugify-each < %s -S -o - | \
; RUN:   opt -strip -strip-dead-prototypes -strip-named-metadata -S -o %t.after
; RUN: diff %t.before %t.after

; Check that stripped IR compares equal before and after applying debugify.
; RUN: opt -O1 < %s | \
; RUN:   opt -strip -strip-dead-prototypes -strip-named-metadata | \
; RUN:   llvm-dis -o %t.before
; RUN: opt -O1 -debugify-each < %s | \
; RUN:   opt -strip -strip-dead-prototypes -strip-named-metadata | \
; RUN:   llvm-dis -o %t.after
; RUN: diff %t.before %t.after

define void @foo(i32 %arg) {
  call i32 asm "bswap $0", "=r,r"(i32 %arg)
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
