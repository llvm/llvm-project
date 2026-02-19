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
; CHECK: L#BB{{[^%]*}} DS 0H
; CHECK-DAG: stg 1,{{.*}}
; CHECK-DAG: st 2,{{.*}}
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = extractvalue { ptr, i32 } %0, 1
  store ptr %1, ptr %ehptr, align 8
  store i32 %2, ptr %ehsel, align 8
  call void @passeh(ptr %1, i32 %2)
  unreachable
}

; Check that offsets to the FD of the personality routine and LSDA are emitted in PPA1
; CHECK: * PPA1 Flags 4
; CHECK: *   Bit 3: 1 = C++ EH block
; CHECK: *   Bit 7: 1 = Name Length and Name
; CHECK:  DC XL1'91'
; CHECK: * Personality routine
; CHECK:  DC XL8'0000000000000020'
; CHECK: * LSDA location
; CHECK:  DC XL8'0000000000000028'
; Check that the exception table is emitted into .lsda section.
; CHECK:  stdin#C CSECT
; CHECK: C_WSA64 CATTR ALIGN(2),FILL(0),NOTEXECUTABLE,RMODE(64),PART(.gcc_excepti
; CHECK:                ion_table.test1)
; CHECK: .gcc_exception_table.test1 XATTR LINKAGE(XPLINK),REFERENCE(DATA),SCOPE(S
; CHECK:                SECTION)
