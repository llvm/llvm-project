; RUN: llc -verify-machineinstrs -mtriple="powerpc-unknown-linux-gnu" -mcpu=ppc64 < %s | FileCheck %s
; PR15286

%va_list = type {i8, i8, i16, ptr, ptr}
declare void @llvm.va_copy(ptr, ptr)

define void @test_vacopy() nounwind {
entry:
	%0 = alloca %va_list
	%1 = alloca %va_list

	call void @llvm.va_copy(ptr %1, ptr %0)

	ret void
}
; CHECK: test_vacopy:
; CHECK-DAG: lwz [[REG1:[0-9]+]], {{.*}}
; CHECK-DAG: lwz [[REG2:[0-9]+]], {{.*}}
; CHECK-DAG: lwz [[REG3:[0-9]+]], {{.*}}
; CHECK-DAG: stw [[REG1]], {{.*}}
; CHECK-DAG: stw [[REG2]], {{.*}}
; CHECK-DAG: stw [[REG3]], {{.*}}
; CHECK: blr
