; RUN: llc < %s -O3 -mtriple=x86_64-linux-unknown -verify-machineinstrs -o %t.s
; RUN: FileCheck --input-file=%t.s %s

; Double-check that we are able to assemble the generated '.s'. A symptom of the
; problem that led to this test is an assembler failure when using
; '-save-temps'. For example:
;  
; > ...s:683:7: error: invalid operand for instruction
; >        addq    $2147483679, %rsp               # imm = 0x8000001F
;
; RUN: llvm-mc -triple x86_64-unknown-unknown %t.s

; Check that the stack update after calling bar gets merged into the second add
; and not the first which is already at the chunk size limit (0x7FFFFFFF).

define void @foo(ptr %rhs) {
; CHECK-LABEL: foo
entry:
  %lhs = alloca [5 x [5 x [3 x [162 x [161 x [161 x double]]]]]], align 16
  store ptr %lhs, ptr %rhs, align 8
  %0 = call i32 @baz()
  call void @bar(i64 0, i64 0, i64 0, i64 0, i64 0, ptr null, ptr %rhs, ptr null, ptr %rhs)
; CHECK: call{{.*}}bar
; CHECK: addq{{.*}}$2147483647, %rsp
; CHECK: addq{{.*}}$372037585, %rsp
  ret void
}

declare void @bar(i64, i64, i64, i64, i64, ptr, ptr, ptr, ptr)

declare i32 @baz()
