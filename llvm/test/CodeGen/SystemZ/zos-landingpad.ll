; RUN: llc < %s -mtriple=s390x-none-zos -mcpu=z10 | FileCheck %s
;
; Ensures that landingpad instructions use the right Exception Pointer
; and Exception Selector registers, and that the exception table is emitted.

declare void @callee()
declare void @passeh(ptr, i32) noreturn
declare i32 @__zos_cxx_personality_v2(...)

define void @test1() uwtable personality ptr @__zos_cxx_personality_v2 {
entry:
  %ehptr = alloca ptr, align 8
  %ehsel = alloca i32, align 8
  invoke void @callee() to label %done unwind label %lpad
done:
  ret void
; Match the return instruction.
; CHECK: b 2(7)
lpad:
  %0 = landingpad { ptr, i32 } cleanup
; The Exception Pointer is %r1; the Exception Selector, %r2.
; CHECK: @BB{{[^%]*}} %lpad
; CHECK-DAG: stg 1, {{.*}}
; CHECK-DAG: st 2, {{.*}}
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = extractvalue { ptr, i32 } %0, 1
  store ptr %1, ptr %ehptr, align 8
  store i32 %2, ptr %ehsel, align 8
  call void @passeh(ptr %1, i32 %2)
  unreachable
}

; Check that offsets to the FD of the personality routine and LSDA are emitted in PPA1
; CHECK: .byte 145 {{.*PPA1 Flags}}
; CHECK: Bit 3: 1 = C++ EH block
; TODO: Emit the value instead of a dummy value.
; CHECK: Personality routine
; CHECK: LSDA location
; Check that the exception table is emitted into .lsda section.
; CHECK: .section ".gcc_exception_table.test1"
; CHECK: GCC_except_table0:
